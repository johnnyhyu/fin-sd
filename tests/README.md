# `tests/`

```bash
python -m pytest tests/
```

110 tests, no GPU and no network. They cover prompt rendering and harmony
encoding, the truncation and fuzzy-alignment logic, the loss slice math, grader
parsing, and GPU reservation.

The suite stubs the model and tokenizer rather than loading weights, so it runs
in seconds on any machine and is the fastest check that a change has not broken
the pipeline's plumbing.
