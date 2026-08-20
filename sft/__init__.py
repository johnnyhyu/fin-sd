"""Standard SFT distillation pipeline.

Distils the teacher (`pipeline.config.HINT_MODEL`, served via OpenRouter) into a
student model by (1) having the teacher solve every problem in
`data/trainingset.json` and (2) supervised-fine-tuning the student on those
prompt+solution pairs.

Run order:  sft.generate_data  →  sft.train   (or `python -m sft.run` for both).
"""
import sys
from pathlib import Path

# Allow `from pipeline import ...` when a submodule is run directly
# (`python sft/train.py`) and the repo root isn't already on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
