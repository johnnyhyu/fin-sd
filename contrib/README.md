# `contrib/`

Work that ships with this repository but produces **no result in the paper**.

## `big-finance-benchmark/`

A substantially modified fork of the Big Finance Harness by Rogo Technologies,
redistributed under its own Apache 2.0 license. It was adapted to run
GPT-OSS-20B — the student this paper post-trains — on an agentic
financial-research benchmark, which required real work against the harmony
format and vLLM's handling of it.

Read [`big-finance-benchmark/CHANGES.md`](big-finance-benchmark/CHANGES.md) for
what was changed and why. Two of the defects fixed there were silent, degrading
scores rather than raising errors, which is worth knowing if you run it.

It is retained as future work, not as a claim. Nothing in the paper depends on it.
