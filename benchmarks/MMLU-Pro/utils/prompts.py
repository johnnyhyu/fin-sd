# MMLU-Pro chain-of-thought prompting.
#
# The system message states the task and the required answer format; `{subject}`
# is filled per question with the record's category (MMLU-Pro mixes 14 domains,
# so the subject is resolved at prompt-build time in inference.py, not here).
# `answer_prefix` is appended after the target question to elicit step-by-step
# reasoning, mirroring the FinanceReasoning `program_prefix` convention.

COT_SYSTEM_INPUT = '''The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'''

COT_ANSWER_PREFIX_INPUT = '''Answer: Let's think step by step.'''


MODEL_PROMPT_DICT = {
    "cot": {
        "system": COT_SYSTEM_INPUT,
        "answer_prefix": COT_ANSWER_PREFIX_INPUT,
        "type": "cot",
    },
}
