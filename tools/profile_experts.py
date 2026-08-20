#!/usr/bin/env python3
"""Profile MoE router activations and hunt for *finance-specialized* experts.

Two modes:

1. Differential (default) — forward a **finance** corpus (``--data``, default the
   FinanceReasoning set) and a **control** corpus (``--control``, default a
   category-balanced MMLU-Pro sample) through the *base* gpt-oss model — no LoRA —
   and compare per-layer / per-expert routing between them. The question it answers
   is: *does finance recruit a distinct subset of experts, or does it route through
   the same experts as everything else?* For each MoE layer it reports the
   Jensen-Shannon divergence between the two selection distributions (how
   differently the layer routes for finance) and lists the experts with the largest
   positive **finance lift** (finance selection share − control selection share).
   If a handful of experts carry large positive lift concentrated in a few layers,
   those are the finance-specialized experts; if the two distributions are ~identical
   everywhere, finance shares generic routing and no specialized set exists.

2. Single-corpus (``--control none``) — the original diagnostic + cache writer:
   forward ``--data`` alone, hook each router, and print a per-layer concentration
   report. The training path consumes that cache when ``PROFILE_EXPERT_ACTIVATIONS=1``
   and ``LORA_TARGET_EXPERTS=1``: ``probing_script._selected_experts_from_profile``
   keeps, per layer, the busiest experts whose cumulative selection mass first reaches
   ``PROFILE_EXPERT_MASS_THRESHOLD`` and freezes the LoRA adapter for the rest (Design
   B — per-expert; since PEFT ranks the fused expert tensor as a whole, the per-expert
   restriction is enforced by grad-masking each non-selected expert's adapter blocks).

This is a read-only diagnostic; it trains nothing and touches no checkpoints. It
loads the unquantized HF model (``config.HF_MODEL_PATH``) on the HF GPUs, so run it
where the training process would run, not against vLLM.

Both corpora accept heterogeneous schemas — items with a ``problem`` field, or with
``context``/``question``/``options`` (FinanceReasoning and MMLU-Pro) — and are
shuffled with ``--seed`` before capping at ``--max-items`` so the control stays
category-balanced.

Examples::

    # Finance (FinanceReasoning) vs control (MMLU-Pro sampled), write the diff:
    python tools/profile_experts.py

    # Preview only, more items, show the identity of the busiest experts:
    python tools/profile_experts.py --max-items 512 --show-experts 6 --no-write

    # Legacy single-corpus cache writer (no differential):
    python tools/profile_experts.py --control none --data data/trainingset.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.utils import logger, setup_logging

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Defaults for the differential: FinanceReasoning as the finance corpus, a
# category-balanced MMLU-Pro sample as the control.
# This repo's own finance corpus, and a category-balanced MMLU-Pro sample as the
# out-of-domain control. launch.py passes --data explicitly, so these apply only
# to a manual invocation.
_FINANCE_DEFAULT = _REPO_ROOT / "data" / "trainingset.json"
_CONTROL_DEFAULT = _REPO_ROOT / "benchmarks" / "MMLU-Pro" / "data" / "MMLU-Pro" / "sampled.json"

_OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_base_model():
    """Materialise the base HF model + tokenizer with NO LoRA attached.

    Mirrors probing_script._load_model_and_tokenizer's load call (dtype, device
    map, per-GPU caps) but stops before get_peft_model — profiling reads the
    frozen router, which the adapter must not perturb.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pipeline import attention
    from pipeline.utils import resolve_max_memory

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(config.HF_DTYPE, torch.bfloat16)

    max_memory = resolve_max_memory(config.HF_MAX_MEMORY,
                                    model_path=config.HF_MODEL_PATH)
    # Same kernel the trainer will load under (gpt-oss autoselects `eager`, whose
    # score matrix is quadratic in sequence length — see pipeline/attention.py).
    attn = attention.resolve_implementation(config.HF_ATTN_IMPLEMENTATION)
    logger.info("Loading BASE model %s (dtype=%s, device_map=%s, attn=%s) for profiling …",
                config.HF_MODEL_PATH, config.HF_DTYPE, config.HF_DEVICE, attn or "auto")
    tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_PATH, use_fast=True)
    load_kwargs = dict(
        torch_dtype=dtype,
        device_map=config.HF_DEVICE,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if attn:
        load_kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(config.HF_MODEL_PATH, **load_kwargs)
    attention.apply_to_loaded(model, attn)
    model.config.use_cache = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _extract_text(item: dict) -> str | None:
    """Pull a prompt string out of one dataset item, schema-agnostic.

    Handles the three schemas we profile: our training set (``problem``),
    FinanceReasoning (``context`` + ``question``), and MMLU-Pro
    (``question`` + multiple-choice ``options``). Returns None when the item
    carries no usable text so the caller can skip it.
    """
    if item.get("problem"):
        return str(item["problem"])
    parts: list[str] = []
    if item.get("context"):
        parts.append(str(item["context"]))
    if item.get("question"):
        parts.append(str(item["question"]))
    opts = item.get("options")
    if isinstance(opts, list) and opts:
        parts.append("\n".join(
            f"{_OPTION_LETTERS[i] if i < len(_OPTION_LETTERS) else i}. {o}"
            for i, o in enumerate(opts)
        ))
    text = "\n".join(parts).strip()
    return text or None


def _load_problems(path: Path, max_items: int, seed: int) -> list[str]:
    """Load, shuffle (fixed seed), and cap prompts from a JSON dataset.

    Shuffling before the cap matters for the control: MMLU-Pro's sampled.json is
    ordered by category, so a naive head-slice would profile only the first few
    subjects instead of a balanced cross-section.
    """
    with open(path) as fh:
        items = json.load(fh)
    if isinstance(items, dict):                      # {id: item} maps → values
        items = list(items.values())
    problems = [t for it in items if isinstance(it, dict) and (t := _extract_text(it))]
    if not problems:
        sys.exit(f"No usable prompts (problem / question fields) in {path}.")
    random.Random(seed).shuffle(problems)
    return problems[:max_items]


# --------------------------------------------------------------------------- #
# Single-corpus concentration report (legacy --control none path)             #
# --------------------------------------------------------------------------- #
def _report(counts: dict[int, list[int]], stats: dict[int, dict],
            top_frac: int, show_experts: int) -> None:
    """Print a per-layer concentration report and an overall summary.

    Columns:
      mass        total selections (constant across layers by construction — a
                  sanity check that every token is routed at every layer).
      used/exp    experts that ever fired / total (dead experts = a prunable tail).
      topN-share  share of selections captured by the N busiest experts — how lossy
                  a per-expert top-N cut would be.
      H_marg      entropy of the aggregate (marginal) routing distribution, bits.
      H_tok       mean per-token routing entropy, bits (dense-logits path only).
      MI          H_marg - H_tok ≈ mutual information I(token; expert). ~0 → the
                  layer is uniform per token (genuinely generic; safe to drop).
                  Large → tokens route sharply but disagree on WHICH experts, so
                  the flat marginal HIDES per-token specialization (do NOT drop).
    """
    have_soft = bool(stats)
    layer_share = []
    hdr = f"\nlayer   mass    used/exp   top{top_frac}-share  H_marg"
    hdr += "   H_tok    MI" if have_soft else ""
    print(hdr)
    print("-" * (len(hdr) + 4))
    for li in sorted(counts):
        c = sorted(counts[li], reverse=True)
        total = sum(c) or 1
        used = sum(1 for x in c if x > 0)
        top_share = sum(c[:top_frac]) / total
        probs = [x / total for x in c if x > 0]
        ent = -sum(p * math.log2(p) for p in probs) if probs else 0.0
        layer_share.append(top_share)
        row = (f"{li:5d}  {total:7d}   {used:3d}/{len(c):<3d}     "
               f"{top_share:6.1%}    {ent:5.2f}")
        if have_soft and li in stats:
            s = stats[li]
            mi = s["h_marg"] - s["h_tok"]
            row += f"   {s['h_tok']:5.2f}  {mi:5.2f}"
        print(row)

    if layer_share:
        avg = sum(layer_share) / len(layer_share)
        print("-" * (len(hdr) + 4))
        print(f"mean top{top_frac}-share across layers: {avg:.1%}  "
              f"(min {min(layer_share):.1%}, max {max(layer_share):.1%})")
        if have_soft:
            mis = [stats[li]["h_marg"] - stats[li]["h_tok"] for li in stats]
            print(f"mean MI(token;expert): {sum(mis)/len(mis):.2f} bits "
                  f"(min {min(mis):.2f}, max {max(mis):.2f})")
            print("→ low MI + high H_marg = near-uniform routing (the cumulative-mass "
                  "cut keeps many experts here);\n  high MI = per-token specialization "
                  "the marginal hides — the busiest experts still carry the mass.")
        else:
            print("→ per-token stats unavailable (dense router logits not "
                  "recoverable); only the marginal is shown.")
        print()

    if show_experts:
        print(f"top-{show_experts} experts per layer (id:share):")
        for li in sorted(counts):
            c = counts[li]
            total = sum(c) or 1
            order = sorted(range(len(c)), key=lambda e: c[e], reverse=True)
            dead = sum(1 for x in c if x == 0)
            top = "  ".join(f"{e:>2d}:{c[e]/total:4.0%}" for e in order[:show_experts])
            tail = f"   (+{dead} dead)" if dead else ""
            print(f"  L{li:2d}  {top}{tail}")
        print()


# --------------------------------------------------------------------------- #
# Differential (finance vs control) analysis                                  #
# --------------------------------------------------------------------------- #
def _dist(counts_list: list[int]) -> list[float]:
    total = sum(counts_list) or 1
    return [c / total for c in counts_list]


def _jsd(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint supports)."""
    def _kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0)
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _diff_report(fin: dict[int, list[int]], ctl: dict[int, list[int]],
                 top_lift: int, show_experts: int) -> dict:
    """Compare finance vs control routing and surface finance-specialized experts.

    For each shared MoE layer, normalises both count vectors to selection-share
    distributions and computes:
      JSD   Jensen-Shannon divergence (bits) between the finance and control
            distributions — how differently the layer routes for finance overall.
      L1    total-variation-style Σ|fin_share − ctl_share| ∈ [0, 2].
      lift  per-expert finance_share − control_share; the busiest positive-lift
            expert per layer is shown inline.

    Then ranks every (layer, expert) by lift to expose the globally most
    finance-specialized experts. Returns a JSON-serialisable payload of the
    per-layer metrics and the top-lift list.
    """
    layers = sorted(set(fin) & set(ctl))
    n_exp = len(next(iter(fin.values()))) if fin else 0

    per_layer = []
    lifts = []                 # (lift, layer, expert, fin_share, ctl_share)
    for li in layers:
        fs, cs = _dist(fin[li]), _dist(ctl[li])
        jsd = _jsd(fs, cs)
        l1 = sum(abs(a - b) for a, b in zip(fs, cs))
        lay_lifts = [(fs[e] - cs[e], li, e, fs[e], cs[e]) for e in range(len(fs))]
        lifts.extend(lay_lifts)
        top = max(lay_lifts, key=lambda x: x[0])
        fin_busy = max(range(len(fs)), key=lambda e: fs[e])
        per_layer.append({
            "layer": li, "jsd": jsd, "l1": l1,
            "top_lift_expert": top[2], "top_lift": top[0],
            "fin_busiest": fin_busy, "fin_busiest_share": fs[fin_busy],
        })

    hdr = "\nlayer   JSD(bits)   L1     finance-busiest     max finance-lift"
    print(hdr)
    print("-" * (len(hdr) + 4))
    for r in per_layer:
        print(f"{r['layer']:5d}    {r['jsd']:6.3f}   {r['l1']:5.3f}   "
              f"e{r['fin_busiest']:>3d} ({r['fin_busiest_share']:5.1%})     "
              f"e{r['top_lift_expert']:>3d}  (+{r['top_lift']:5.1%})")

    jsds = [r["jsd"] for r in per_layer]
    mean_jsd = sum(jsds) / len(jsds) if jsds else 0.0
    print("-" * (len(hdr) + 4))
    print(f"mean JSD across {len(layers)} layers: {mean_jsd:.3f} bits  "
          f"(min {min(jsds):.3f}, max {max(jsds):.3f})")

    # Global ranking of the most finance-specialized (layer, expert) cells.
    lifts.sort(reverse=True)
    print(f"\ntop-{top_lift} finance-specialized experts (layer.expert  "
          f"fin-share vs ctl-share  = lift):")
    top_cells = []
    for lift, li, e, fsh, csh in lifts[:top_lift]:
        ratio = (fsh + 1e-9) / (csh + 1e-9)
        print(f"  L{li:2d}.e{e:<3d}   {fsh:6.1%}  vs {csh:6.1%}   "
              f"= +{lift:5.1%}   (×{ratio:4.1f})")
        top_cells.append({"layer": li, "expert": e, "fin_share": fsh,
                          "ctl_share": csh, "lift": lift, "ratio": ratio})

    # Verdict. Two independent signals: how much layers diverge (mean JSD), and
    # how concentrated the finance advantage is in a few experts (top-lift mass).
    top10_mass = sum(x[0] for x in lifts[:10] if x[0] > 0)
    print()
    strong = mean_jsd > 0.05 or top10_mass > 0.30
    if strong:
        print(f"→ VERDICT: finance recruits a distinguishable expert subset "
              f"(mean JSD {mean_jsd:.3f} bits; top-10 lift mass {top10_mass:.1%}). "
              f"The (layer.expert) cells above are the finance-leaning experts; "
              f"the highest-JSD layers are where routing diverges most.")
    else:
        print(f"→ VERDICT: routing is largely shared with the control "
              f"(mean JSD {mean_jsd:.3f} bits; top-10 lift mass {top10_mass:.1%}). "
              f"No strongly finance-specialized expert set — finance flows through "
              f"generic experts.")
    print()

    return {
        "num_experts": n_exp,
        "num_layers": len(layers),
        "mean_jsd": mean_jsd,
        "per_layer": per_layer,
        "top_lift": top_cells,
        "top10_lift_mass": top10_mass,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(_FINANCE_DEFAULT),
                    help="Finance corpus (default: data/trainingset.json). "
                         "Items may use 'problem' or 'context'/'question'/'options'.")
    ap.add_argument("--control", default=str(_CONTROL_DEFAULT),
                    help="Control corpus for the differential (default: MMLU-Pro "
                         "sampled.json). Pass 'none' for the legacy single-corpus "
                         "concentration report + EXPERT_PROFILE_PATH cache.")
    ap.add_argument("--out", default=None,
                    help="Where to write the result cache. Default: "
                         "data/expert_finance_diff.json in differential mode, "
                         "config.EXPERT_PROFILE_PATH in single-corpus mode.")
    ap.add_argument("--max-items", type=int, default=config.PROFILE_MAX_ITEMS,
                    help="Max items per corpus (router stats saturate quickly).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Shuffle seed applied before --max-items capping.")
    ap.add_argument("--top-frac", type=int, default=8,
                    help="Single-corpus mode: report the share captured by each "
                         "layer's top-N experts.")
    ap.add_argument("--top-lift", type=int, default=30, metavar="N",
                    help="Differential mode: list the N most finance-specialized "
                         "(layer, expert) cells.")
    ap.add_argument("--show-experts", type=int, default=0, metavar="N",
                    help="Single-corpus mode: also list each layer's top-N expert "
                         "ids and shares (the identity view). 0 = off.")
    ap.add_argument("--no-write", action="store_true",
                    help="Print the report only; do not write the cache.")
    args = ap.parse_args()

    setup_logging()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Finance dataset not found: {data_path}")
    fin_problems = _load_problems(data_path, args.max_items, args.seed)

    differential = args.control.lower() != "none"
    ctl_path = None
    if differential:
        ctl_path = Path(args.control)
        if not ctl_path.exists():
            sys.exit(f"Control dataset not found: {ctl_path}")
        ctl_problems = _load_problems(ctl_path, args.max_items, args.seed)
        logger.info("Differential: finance=%d item(s) from %s vs control=%d item(s) "
                    "from %s.", len(fin_problems), data_path.name,
                    len(ctl_problems), ctl_path.name)
    else:
        logger.info("Single-corpus profile over %d item(s) from %s.",
                    len(fin_problems), data_path.name)

    from pipeline.probing_script import profile_expert_activations

    model, tokenizer = _load_base_model()
    fin_counts, fin_stats = profile_expert_activations(
        model, tokenizer, fin_problems, max_items=args.max_items
    )

    if not differential:
        _report(fin_counts, fin_stats, args.top_frac, args.show_experts)
        if args.no_write:
            logger.info("--no-write: cache not written.")
            return
        out = Path(args.out) if args.out else Path(config.EXPERT_PROFILE_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": config.HF_MODEL_PATH,
            "data": str(data_path),
            "n_items": len(fin_problems),
            "num_experts": len(next(iter(fin_counts.values()))) if fin_counts else 0,
            "counts": {str(li): c for li, c in fin_counts.items()},
            "stats": {str(li): s for li, s in fin_stats.items()},
        }
        with open(out, "w") as fh:
            json.dump(payload, fh)
        logger.info("Wrote expert profile → %s (%d layers). Set "
                    "PROFILE_EXPERT_ACTIVATIONS=1 to use it on the next "
                    "expert-LoRA run.", out, len(fin_counts))
        return

    ctl_counts, _ = profile_expert_activations(
        model, tokenizer, ctl_problems, max_items=args.max_items
    )
    diff = _diff_report(fin_counts, ctl_counts, args.top_lift, args.show_experts)

    if args.no_write:
        logger.info("--no-write: cache not written.")
        return

    out = Path(args.out) if args.out else _REPO_ROOT / "data" / "expert_finance_diff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": config.HF_MODEL_PATH,
        "finance_data": str(data_path),
        "control_data": str(ctl_path),
        "n_finance": len(fin_problems),
        "n_control": len(ctl_problems),
        "seed": args.seed,
        "finance_counts": {str(li): c for li, c in fin_counts.items()},
        "control_counts": {str(li): c for li, c in ctl_counts.items()},
        **diff,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh)
    logger.info("Wrote finance-vs-control expert diff → %s (%d layers, mean JSD "
                "%.3f bits).", out, diff["num_layers"], diff["mean_jsd"])


if __name__ == "__main__":
    main()
