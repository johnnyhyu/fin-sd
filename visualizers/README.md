# `visualizers/`

Static HTML viewers for run output. Serve the repo root and open them:

```bash
python -m http.server
# then http://localhost:8000/visualizers/
```

| | |
|---|---|
| `reasoning_visualizer.html` | Browse student trajectories, the localized error, and the truncation point. Reads `../data/reasoning_log.jsonl`, written during a training run. |
| `feedback_visualizer.html` | Corpus QA review. Self-contained — data is embedded. |

`reasoning_log.jsonl` is run output and is gitignored, so the first viewer is
blank until you have trained. Nothing else depends on these; they are for
looking at what the pipeline did.
