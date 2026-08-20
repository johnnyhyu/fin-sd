# Modifications to the Big Finance Harness

This directory vendors a **substantially modified** copy of the Big Finance
Harness by Rogo Technologies, redistributed under its own Apache License 2.0
(see `LICENSE`). This file records the modifications, as Apache-2.0 section 4(b)
requires.

**Nothing in this directory produces a result in the ICAIF paper.** It is
retained as future work: the harness was adapted to run GPT-OSS-20B so the same
student the paper post-trains can be evaluated on an agentic financial-research
benchmark. Read it as work in progress, not as a claim.

## Why the changes were needed

Upstream targets hosted API models through LiteLLM. GPT-OSS speaks *harmony*, a
channel-structured format, and is served here by a local vLLM instance. Most of
the work below is the collision between those two facts. Two of the defects were
silent — they degraded scores rather than raising errors — which is the main
reason this file exists.

## Changes

### `big_finance_harness/config.py` — harmony stop tokens

vLLM registers `<|call|>` (200012) and `<|return|>` (200002) as default stop
tokens for gpt-oss, but its chat-completions path passes the *request's*
`stop_token_ids` (defaulting to `[]`) straight into `SamplingParams` without
merging the server defaults. A plain `/v1/chat/completions` call therefore runs
past `<|call|>`, the model begins another harmony message, and vLLM's own parser
raises `HarmonyError: Unexpected token 12606 while expecting start token 200006`
— a 500 on *every* tool-enabled request, which is every request this harness
makes. `VLLM_HARMONY_EXTRA_BODY` sends the stop tokens explicitly. The ids exist
only in the o200k_harmony vocabulary, so they are inert for a local server
hosting anything else.

### `big_finance_harness/tools/final_answer.py` — terminal tool rename

gpt-oss emits a tool call as `<|channel|>final_answer<|message|>…`, writing the
tool's name where harmony expects a channel. vLLM's parser knows only
`analysis`, `commentary` and `final`, so it drops the message whole: the turn
arrives with no content and no tool call, the agent loop reads an empty turn,
nudges, and eventually abandons the question. **Measured on the 50-question
public subset this cost gpt-oss-20b roughly a third of its runs** — a harness
artifact that scores as model failure.

The tool name is now configurable via `BFH_TERMINAL_TOOL_NAME` and interpolated
into the system prompt and the empty-turn nudges rather than written out in
each, so they cannot drift from it. The default stays `final_answer` so
previously recorded runs remain comparable; set `submit_answer` for any gpt-oss
arm, and for whatever arm it is compared against.

### `big_finance_harness/models/base.py` — harmony message layer

- Parse a harmony assistant turn, which is a sequence of channelled messages,
  into the harness's single-text-plus-tool-calls shape. Without this the harness
  records `assistant_text=''` for every step of a gpt-oss trace.
- Map local checkpoint routes (`local:epoch_1`) onto LiteLLM's
  `hosted_vllm/epoch_1` form so a served epoch is addressable like any model.
- Repair malformed tool names the model emits (`python_exec.`, `"fetch_url"`,
  `search?`, `functions?`) rather than failing the step.
- Forward `reasoning_effort` for reasoning models, and raise the per-request
  timeout to cover high-effort reasoning on hard questions.

### `big_finance_harness/agent.py` — loop robustness

Treat `HarmonyError` responses as retryable, allow a bounded number of empty
turns before abandoning a question, and name the available tools in the error
returned for an unknown tool, which gpt-oss-20b traces showed the model needed.

### `config/config.yaml` — run configuration

Restructured to mirror `benchmarks/FinanceReasoning/config/config.yaml` and
`benchmarks/MMLU-Pro/config/config.yaml`, so all four vendored benchmarks in
this repository are driven by one config shape. Adds a `local-model1` route for
the served checkpoint, documents the `vllm serve` flags the harness needs
(`--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser
openai_gptoss`), and sets the harmony `extra_body` default.

### `tests/`

Added `test_contamination.py`, `test_preflight.py` and `test_trace_summary.py`.

## Unchanged

The ReAct loop's structure, the four public tools, the grader and rubric
handling, the contamination guard's policy, the dataset, and the datasheet are
upstream's. The public 50-question subset in `data/` is licensed separately
under CC BY 4.0; see `data/LICENSE-DATA`.
