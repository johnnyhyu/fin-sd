#!/usr/bin/env python3
"""
FinTrust inference: run every dataset in ONE process under a single global
concurrency cap. Companion to evaluation.py, and driven the same way as
FinanceReasoning — everything is read from config/config.yaml:

    python inference.py --config config/config.yaml
    python inference.py --list                     # print dataset names

Every dataset's api_call.py exposes:

    get_jobs(input_path, model_key, sem, *, system_prompt_type=None)
        -> (tasks, writer)

where `tasks` are per-item coroutines that gate each API call on the shared
`sem`, and `writer(results)` writes that dataset's output JSON. This launcher
creates ONE asyncio.Semaphore(run.global_parallel), builds the jobs for every
dataset, and awaits them all together. Because a single semaphore gates every
request across all datasets, the true number of in-flight calls never exceeds
the cap, and whenever a slot frees it is immediately taken by whichever dataset
still has work — so all slots stay busy until everything drains. There are no
phases and no per-dataset subprocesses.

Model, global cap, and dataset selection come from the config's `run:` section
(run.model / run.global_parallel / run.only). Privacy is evaluated under three
system-prompt conditions (without / implicit / explicit), each written to its
own file; all three run unless run.system_prompt_type pins a single condition.
"""

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from api_utils import (  # noqa: E402
    report_failures,
    GLOBAL_PARALLEL,
    DEFAULT_CONFIG_PATH,
    load_config,
    set_catalog_from_config,
)

# name -> (api_call.py script, input dataset). Paths are relative to REPO_ROOT.
DATASETS = {
    "fairness/industry-level_data": (
        "fairness/industry-level_data/api_call.py",
        "fairness/industry-level_data/fairness_industry_level_evaluation_100.json",
    ),
    "fairness/personal-level_data": (
        "fairness/personal-level_data/api_call.py",
        "fairness/personal-level_data/fairness_personal_level_evaluation_100.jsonl",
    ),
    "knowledge-discovery": (
        "knowledge-discovery/api_call.py",
        "knowledge-discovery/knowledge-discovery_evaluation_100.json",
    ),
    "privacy": (
        "privacy/api_call.py",
        "privacy/privacy_evaluation_100.json",
    ),
    "robustness": (
        "robustness/api_call.py",
        "robustness/Robustness_evaluation_20.json",
    ),
    "safety/deep-inception": (
        "safety/deep-inception/api_call.py",
        "safety/deep-inception/deep_inception_evaluation_100.json",
    ),
    "safety/in-context attack": (
        "safety/in-context attack/api_call.py",
        "safety/in-context attack/in_context_evaluation_100.json",
    ),
    "safety/language_barrier": (
        "safety/language_barrier/api_call.py",
        "safety/language_barrier/language_barrier_evaluation_100.json",
    ),
    "safety/plain_question": (
        "safety/plain_question/api_call.py",
        "safety/plain_question/plain_attack_evaluation_100.json",
    ),
    "transparency": (
        "transparency/api_call.py",
        "transparency/transparency_evaluation_100.json",
    ),
    "trustfulness/hallucination_certainty": (
        "trustfulness/hallucination_certainty/api_call.py",
        "trustfulness/hallucination_certainty/hallucination_certainty_evaluation_100.jsonl",
    ),
    "trustfulness/number_informativeness": (
        "trustfulness/number_informativeness/api_call.py",
        "trustfulness/number_informativeness/number_informativeness_evaluation_100.jsonl",
    ),
}

# Privacy is run once per system-prompt condition (each writes its own file).
PRIVACY_CONDITIONS = ("without", "implicit", "explicit")


def load_get_jobs(name, script):
    """Import a dataset's api_call.py and return its get_jobs function.

    Each module is loaded under a unique name (they are all called "api_call")
    from its file path — several live in directories that aren't importable as
    packages (spaces/hyphens), so spec_from_file_location is used directly.
    """
    mod_name = "fintrust_ds_" + name.replace("/", "_").replace(" ", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, str(REPO_ROOT / script))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "get_jobs"):
        raise AttributeError(f"{script} does not expose get_jobs()")
    return module.get_jobs


def build_plan(names, model_key, sem, pin_sp_type):
    """Return (all_tasks, segments).

    segments is a list of (display_name, start, count, writer): where this
    dataset's results live in the concatenated results list, and how to write
    them. Privacy fans out to one segment per system-prompt condition.
    """
    all_tasks = []
    segments = []
    for name in names:
        script, input_path = DATASETS[name]
        get_jobs = load_get_jobs(name, script)

        if name == "privacy" and not pin_sp_type:
            conditions = PRIVACY_CONDITIONS
        elif name == "privacy":
            conditions = (pin_sp_type,)
        else:
            conditions = (None,)

        for cond in conditions:
            display = f"privacy[{cond}]" if name == "privacy" else name
            print(f"\n===== preparing {display} =====")
            tasks, writer = get_jobs(input_path, model_key, sem, system_prompt_type=cond)
            segments.append((display, len(all_tasks), len(tasks), writer))
            all_tasks.extend(tasks)
    return all_tasks, segments


async def run(names, model_key, global_parallel, pin_sp_type):
    sem = asyncio.Semaphore(global_parallel)
    all_tasks, segments = build_plan(names, model_key, sem, pin_sp_type)

    print(f"\nLaunching {len(all_tasks)} items across {len(segments)} dataset run(s) "
          f"under a shared global cap of {global_parallel} concurrent requests.\n")
    results = await tqdm_asyncio.gather(*all_tasks, desc="All datasets", total=len(all_tasks))

    print("\n===== summary =====")
    any_failed = False
    for display, start, count, writer in segments:
        segment_results = results[start:start + count]
        out_path = writer(segment_results)
        failed = report_failures(segment_results, out_path, exit_on_failure=False)
        mark = "ERR" if failed else "ok "
        if failed:
            any_failed = True
        print(f"  [{mark}] {display}  ->  {out_path}")
    return 1 if any_failed else 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all FinTrust datasets in one process under a global cap. "
                    "Everything (model, parallelism, dataset selection) is read from "
                    "the --config YAML, just like FinanceReasoning.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Run config YAML (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--list", action="store_true", help="List dataset names and exit.")
    args = parser.parse_args()

    if args.list:
        for name in DATASETS:
            print(name)
        return 0

    # config/config.yaml is the source of truth. Point every process (including
    # any subprocess-imported module) at this exact file, then (re)build the
    # model catalog from it.
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    os.environ["FINTRUST_CONFIG"] = str(config_path)
    config = load_config(config_path)
    set_catalog_from_config(config)

    run_cfg = config.get("run", {})
    model_key = run_cfg.get("model", "gpt-4.1-mini")
    global_parallel = int(run_cfg.get("global_parallel") or GLOBAL_PARALLEL)
    pin_sp_type = run_cfg.get("system_prompt_type")

    only = run_cfg.get("only")
    names = only if only else list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"Unknown dataset name(s) in run.only: {', '.join(unknown)}. Use --list to see valid names.")

    return asyncio.run(run(names, model_key, global_parallel, pin_sp_type))


if __name__ == "__main__":
    raise SystemExit(main())
