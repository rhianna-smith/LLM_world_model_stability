INSTRUCTION_TEMPLATE = """
Answer the question in the INPUT.

Choose exactly ONE option.

Rules:
- The answer must EXACTLY match one of the provided options word-for-word
- The answer should be stated as the full description of one of the options
- Do not paraphrase
- Do not shorten the option
- Do not add punctuation to the answer
- Choose only one answer

Example:

INPUT:
A person is low income. Based on this information, would this person most likely choose a Nikon camera that costs 52.74 dollars or a Fujifilm camera that costs 52.97 dollars?

The answer is a Nikon camera that costs 52.74 dollars because a low income person would likely prefer to save money.
"""

REVERSE_INSTRUCTION_TEMPLATE = """
Answer the question in the INPUT.

Choose exactly ONE option.

Rules:
- The answer must EXACTLY match one of the provided options word-for-word
- The answer should be stated as the full description of one of the options
- Do not paraphrase
- Do not shorten the option
- Do not add punctuation to the answer
- Choose only one answer

Example:

INPUT:
A person is more likely to choose a Fujifilm camera that costs 52.97 dollars rather than a Nikon camera that costs 52.74 dollars. Based on this information, which of the following options is more likely to be true: this person is low income or this person is not low income?

The answer is this person is not low income because choosing the more expensive option suggests they may have more disposable income.
"""
