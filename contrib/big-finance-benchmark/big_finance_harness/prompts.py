"""System prompt for the Big Finance lightweight harness.

Kept deliberately short. The contract with the model is: you have these tools, you may call
them in parallel or in sequence, terminate by calling the terminal tool. The prompt does not
inject any finance-specific guidance, playbooks, or chain-of-thought instructions; that
would be a harness contribution that confounds the model evaluation.

The terminal tool's name is interpolated from `TERMINAL_TOOL_NAME` rather than written
out, so the prompt cannot name a tool that isn't the one registered — it is renameable
via BFH_TERMINAL_TOOL_NAME (see tools/final_answer.py for why it has to be).
"""

from big_finance_harness.tools.final_answer import TERMINAL_TOOL_NAME

SYSTEM_PROMPT = f"""\
You are answering finance research questions for an academic benchmark.

You have access to a small set of tools: web search, SEC EDGAR filings, URL fetching with
optional in-document retrieval, a sandboxed Python REPL, and a final-answer tool. Use them
to investigate primary sources, perform calculations, and arrive at a single defensible
answer.

When you are confident in your answer, call the `{TERMINAL_TOOL_NAME}` tool with a short
response. Reference answers are typically a single number with units; match that format
when possible. If the question cannot be answered from the available sources, say so
explicitly in `{TERMINAL_TOOL_NAME}`.
"""
