#!/usr/bin/env python3
"""Re-run only the FAILED items (those containing an "ERROR:" string) in an
existing api_call.py output JSON, in place. Recovers runs that hit OpenRouter
rate limits / connection errors without redoing the whole batch.

Run it from the repo root, the same way you run the api_call.py scripts
(the target module does a relative load_dotenv):

    MODEL_KEY=gpt-oss-120b python retry_errors.py safety/language_barrier <output.json> [<output2.json> ...]

The module dir (or its api_call.py path) is the first argument; one or more
output JSON files follow. MODEL_KEY is taken from the environment, else inferred
from the output filename. A .bak copy is written before the file is overwritten.
"""
import sys
import os
import json
import shutil
import asyncio
import inspect
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_utils import contains_error
from tqdm.asyncio import tqdm_asyncio


def load_module(arg):
    path = os.path.join(arg, "api_call.py") if os.path.isdir(arg) else arg
    if not os.path.exists(path):
        sys.exit(f"No api_call.py found at: {path}")
    spec = importlib.util.spec_from_file_location("target_api_call", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # __name__ != '__main__', so main() does NOT run
    return mod


def find_process_fn(mod):
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("process_one") and inspect.iscoroutinefunction(getattr(mod, n))]
    if len(fns) != 1:
        sys.exit(f"Expected exactly one module-level process_one* coroutine, found {len(fns)}.")
    return fns[0]


def resolve_model(mod, out_path):
    catalog = mod.MODEL_CATALOG
    key = os.environ.get("MODEL_KEY") or getattr(mod, "MODEL_KEY", None)
    if key not in catalog:
        for k in catalog:                 # infer from filename, e.g. ....gpt-oss-120b....json
            if f".{k}." in os.path.basename(out_path):
                key = k
                break
    if key not in catalog:
        sys.exit(f"Could not determine MODEL_KEY. Set MODEL_KEY to one of: {list(catalog)}")
    info = catalog[key]
    return key, info["client"], info["model"]


def make_redo(proc, mod, client, model, sem, records):
    """Return an async fn redo(i) that re-runs record i and stores the fresh
    result back into records[i], dispatching on the process fn's signature."""
    params = list(inspect.signature(proc).parameters)

    async def redo(i):
        rec = records[i]
        if "gpt4mini_client" in params:                       # language_barrier
            g = mod.MODEL_CATALOG["gpt-4.1-mini"]
            prompt_dict = {
                "translated": {"hau_Latn": rec.get("low_resource_language_attack", "")},
                "plain_attack": rec.get("plain_attack"),
            }
            new = await proc(rec.get("topic", "default"), prompt_dict,
                             client, model, sem, g["client"], g["model"])
        elif params[:2] == ["topic", "prompt"]:               # plain_question
            new = await proc(rec.get("topic", "default"), rec.get("plain_attack", ""),
                             client, model, sem)
        elif params[:2] == ["topic", "item"]:                 # in-context attack
            new = await proc(rec.get("topic", "default"), rec, client, model, sem)
        elif params[0] == "item":                             # item-based modules
            args = [rec, client, model, sem]
            if "idx" in params:
                args.append(i)
            if "items" in params:
                args.append(records)
            new = await proc(*args)
        else:
            raise RuntimeError(f"Unrecognized process signature: {params}")
        records[i] = new

    return redo


async def salvage(mod, proc, out_path):
    key, client, model = resolve_model(mod, out_path)
    with open(out_path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        sys.exit(f"{out_path}: expected a JSON list of records.")

    failed = [i for i, r in enumerate(records) if contains_error(r)]
    if not failed:
        print(f"{out_path}: no ERROR items — nothing to do.")
        return
    print(f"{out_path}: re-running {len(failed)}/{len(records)} failed items with model '{key}'...")

    sem = asyncio.Semaphore(getattr(mod, "MAX_PARALLEL", 6))
    redo = make_redo(proc, mod, client, model, sem, records)
    await tqdm_asyncio.gather(*[redo(i) for i in failed], desc="Retrying", total=len(failed))

    still = [i for i in failed if contains_error(records[i])]
    if not os.path.exists(out_path + ".bak"):
        shutil.copy2(out_path, out_path + ".bak")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    recovered = len(failed) - len(still)
    print(f"{out_path}: recovered {recovered}/{len(failed)}; {len(still)} still failing. "
          f"(backup at {out_path}.bak)")


async def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mod = load_module(sys.argv[1])
    proc = find_process_fn(mod)
    for out_path in sys.argv[2:]:
        await salvage(mod, proc, out_path)


if __name__ == "__main__":
    asyncio.run(main())
