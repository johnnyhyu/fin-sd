# `tools/`

| | |
|---|---|
| `serve_checkpoint.py` | Bring vLLM up on one epoch's checkpoint, merging its adapter first when the merged copy has been pruned. **This is how you evaluate a trained epoch.** |
| `serve_epoch.py` | The non-expert-LoRA equivalent: serve the base model and hot-load the epoch's adapter on top. |
| `profile_experts.py` | Profile MoE router activations to choose which experts the expert-LoRA path adapts. `launch.py` runs this automatically when the cache is missing or was built for a different model. |

```bash
python tools/serve_checkpoint.py --epoch 4
python tools/serve_checkpoint.py --source opsd --run-id 20260730-1200 --epoch 2
```

Stop the training pipeline first — it owns the vLLM port. See
[../docs/RUNNING.md](../docs/RUNNING.md).
