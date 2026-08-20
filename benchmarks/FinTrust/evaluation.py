#!/usr/bin/env python3
"""
FinTrust evaluation: post-process every dataset in parallel. Driven the same way
as FinanceReasoning — everything is read from config/config.yaml:

    python evaluation.py --config config/config.yaml
    python evaluation.py --list                     # print dataset names

This is the companion to inference.py. Where inference.py runs each category's
api_call.py (producing a "<input>.<model>[.<system_prompt>].<suffix>.json"
response file), this launcher runs each category's postprocess script over
that response file to produce the judged / scored / statistics output.

Each postprocess script reads its input path from the RESPONSE_JSON_PATH
environment variable (falling back to its own hard-coded default). This
launcher reconstructs the exact response filename that inference.py / api_call.py
would have written for the configured postprocess.model (and privacy condition),
injects it as RESPONSE_JSON_PATH, and waits for all scripts to finish.

Model, dataset selection, concurrency, and privacy condition come from the
config's `postprocess:` section. Privacy is evaluated under three system-prompt
conditions (without / implicit / explicit), each with its own response file; all
three are post-processed unless postprocess.system_prompt_type pins one.

Per-dataset stdout/stderr is written to logs/<name>.postprocess.log and key
lines are echoed live with a [name] prefix.
"""

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import dotenv_values

from api_utils import DEFAULT_CONFIG_PATH, load_config

REPO_ROOT = Path(__file__).resolve().parent

# name -> (postprocess script, input dataset, api_call output suffix,
#          needs_system_prompt). Paths are relative to REPO_ROOT.
#
# The response file that api_call.py writes is:
#   "<splitext(input)[0]>.<MODEL_KEY>[.<SYSTEM_PROMPT_TYPE>].<suffix>.json"
# The SYSTEM_PROMPT_TYPE segment is only present where needs_system_prompt.
DATASETS = {
    "fairness/industry-level_data": (
        "fairness/industry-level_data/postprocess_response.py",
        "fairness/industry-level_data/fairness_industry_level_evaluation_100.json",
        "llm_company-level-fairness_output",
        False,
    ),
    "fairness/personal-level_data": (
        "fairness/personal-level_data/postprocess_response.py",
        "fairness/personal-level_data/fairness_personal_level_evaluation_100.jsonl",
        "llm_personal-level-fairness_output",
        False,
    ),
    "knowledge-discovery": (
        "knowledge-discovery/postprocess_response.py",
        "knowledge-discovery/knowledge-discovery_evaluation_100.json",
        "output",
        False,
    ),
    "privacy": (
        "privacy/postprocess_response.py",
        "privacy/privacy_evaluation_100.json",
        "output",
        True,
    ),
    "robustness": (
        "robustness/postprocess_response.py",
        "robustness/Robustness_evaluation_20.json",
        "llm6output",
        False,
    ),
    "safety/deep-inception": (
        "safety/deep-inception/postprocess_response.py",
        "safety/deep-inception/deep_inception_evaluation_100.json",
        "llm_deep_inception_attack_output",
        False,
    ),
    "safety/in-context attack": (
        "safety/in-context attack/postprocess_response.py",
        "safety/in-context attack/in_context_evaluation_100.json",
        "llm_plain_attack_output",
        False,
    ),
    "safety/language_barrier": (
        "safety/language_barrier/postprocess_response.py",
        "safety/language_barrier/language_barrier_evaluation_100.json",
        "llm_low_resource_attack_output",
        False,
    ),
    "safety/plain_question": (
        "safety/plain_question/postprocess_response.py",
        "safety/plain_question/plain_attack_evaluation_100.json",
        "llm_plain_attack_output",
        False,
    ),
    "transparency": (
        "transparency/postprocess_response.py",
        "transparency/transparency_evaluation_100.json",
        "llm_investment_suggestion_output",
        False,
    ),
    "trustfulness/hallucination_certainty": (
        "trustfulness/hallucination_certainty/postprocess_response.py",
        "trustfulness/hallucination_certainty/hallucination_certainty_evaluation_100.jsonl",
        "output",
        False,
    ),
    "trustfulness/number_informativeness": (
        "trustfulness/number_informativeness/postprocess_response.py",
        "trustfulness/number_informativeness/number_informativeness_evaluation_100.jsonl",
        "output",
        False,
    ),
}

# Privacy is post-processed once per system-prompt condition (each has its own
# response file). Kept in sync with inference.PRIVACY_CONDITIONS.
PRIVACY_CONDITIONS = ("without", "implicit", "explicit")

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def response_path(input_path, suffix, needs_system_prompt, model_key, system_prompt_type):
    """Reconstruct the response filename that api_call.py wrote for a dataset."""
    base = os.path.splitext(input_path)[0]
    parts = [base, model_key]
    if needs_system_prompt:
        parts.append(system_prompt_type)
    parts.append(suffix)
    return ".".join(parts) + ".json"


def run_dataset(name, script, input_json, base_env, log_dir, log_name=None):
    """Run one postprocess script as a subprocess. Returns (name, returncode).

    `log_name` overrides the log filename so concurrent runs of the same
    dataset (e.g. privacy's three conditions) don't clobber each other's logs.
    """
    env = dict(base_env)
    env["RESPONSE_JSON_PATH"] = input_json

    log_file = log_dir / ((log_name or name.replace("/", "__")) + ".postprocess.log")

    if not (REPO_ROOT / input_json).exists():
        log(f"[{name}] WARNING: response file not found -> {input_json} "
            f"(run inference.py first, or check the model/privacy condition in config)")

    log(f"[{name}] starting  ->  {input_json}")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            stripped = line.rstrip()
            # Echo the informative lines live; skip the noisy tqdm redraws.
            if stripped and ("Results written" in stripped
                             or "Results saved" in stripped
                             or stripped.startswith("Loaded")
                             or stripped.startswith("Answered")
                             or stripped.startswith("Average")
                             or "ERROR" in stripped
                             or "Error" in stripped
                             or "Retry" in stripped):
                log(f"[{name}] {stripped}")
        rc = proc.wait()

    status = "done" if rc == 0 else f"FAILED (exit {rc})"
    log(f"[{name}] {status}  (log: {log_file.relative_to(REPO_ROOT)})")
    return name, rc


def main():
    parser = argparse.ArgumentParser(
        description="Post-process all FinTrust datasets in parallel. Model, dataset "
                    "selection, and privacy condition are read from the --config YAML, "
                    "just like FinanceReasoning.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Run config YAML (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--list", action="store_true", help="List dataset names and exit.")
    args = parser.parse_args()

    if args.list:
        for name in DATASETS:
            print(name)
        return 0

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    config = load_config(config_path)
    pp_cfg = config.get("postprocess", {})

    only = pp_cfg.get("only")
    selected = DATASETS
    if only:
        unknown = [n for n in only if n not in DATASETS]
        if unknown:
            parser.error(f"Unknown dataset name(s) in postprocess.only: {', '.join(unknown)}. Use --list to see valid names.")
        selected = {n: DATASETS[n] for n in only}

    # Load the root .env (OPENROUTER_API_KEY, ...) and pass it to every
    # subprocess, mirroring inference.py. Explicit shell env wins over the .env file.
    base_env = dict(os.environ)
    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        for k, v in dotenv_values(root_env).items():
            if v is not None:
                base_env.setdefault(k, v)

    # RESPONSE_JSON_PATH is set per-dataset below, so drop any inherited value.
    base_env.pop("RESPONSE_JSON_PATH", None)

    # config/config.yaml is the source of truth: which model's responses to
    # post-process, and the privacy condition. Point subprocesses at this exact
    # config file and inject MODEL_KEY so any judge model they call resolves the
    # same catalog.
    model_key = pp_cfg.get("model") or "gpt-4.1-mini"
    base_env["FINTRUST_CONFIG"] = str(config_path)
    base_env["MODEL_KEY"] = model_key

    # postprocess.system_prompt_type: a concrete condition pins privacy to it;
    # null (unset) fans privacy out across all three conditions.
    sp_cfg = pp_cfg.get("system_prompt_type")
    pin_privacy = sp_cfg is not None
    system_prompt_type = (sp_cfg or "implicit").lower().strip()
    max_concurrent = int(pp_cfg.get("max_concurrent") or 0)

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # Expand into concrete jobs. Privacy fans out to one job per system-prompt
    # condition unless config pinned one via postprocess.system_prompt_type.
    # Each job: (display_name, log_name, script, response_path).
    jobs = []
    for name, (script, input_path, suffix, needs_sp) in selected.items():
        if name == "privacy" and not pin_privacy:
            for cond in PRIVACY_CONDITIONS:
                resp = response_path(input_path, suffix, needs_sp, model_key, cond)
                jobs.append((f"privacy[{cond}]", f"privacy__{cond}", script, resp))
        else:
            resp = response_path(input_path, suffix, needs_sp, model_key, system_prompt_type)
            jobs.append((name, name.replace("/", "__"), script, resp))

    workers = max_concurrent if max_concurrent > 0 else len(jobs)
    log(f"Post-processing {len(jobs)} job(s) across {len(selected)} dataset(s) "
        f"for model '{model_key}', up to {workers} at a time.\n")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_dataset, display, script, resp, base_env, log_dir,
                        log_name=log_name): display
            for display, log_name, script, resp in jobs
        }
        for fut in as_completed(futures):
            name, rc = fut.result()
            results[name] = rc

    log("\n===== summary =====")
    failed = [n for n, rc in results.items() if rc != 0]
    for display, *_ in jobs:
        mark = "ok " if results.get(display) == 0 else "ERR"
        log(f"  [{mark}] {display}")
    if failed:
        log(f"\n{len(failed)} job(s) failed: {', '.join(failed)}")

    # Read back the judged/scored files and print one headline metric each.
    try:
        from summarize_results import summarize
        summarize(model_key, system_prompt_type, names=list(selected), repo_root=REPO_ROOT, log=log)
    except Exception as e:  # noqa: BLE001 - a summary hiccup shouldn't fail the run
        log(f"\n(results summary skipped: {e})")

    if failed:
        return 1
    log(f"\nAll {len(jobs)} job(s) across {len(selected)} dataset(s) post-processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
