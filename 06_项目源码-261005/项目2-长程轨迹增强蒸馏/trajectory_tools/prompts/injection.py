"""过程监督注入的提示词：英文第一人称状态回顾，只陈述事实与依据，不给操作指令。"""

INJECTION_PROMPT = """You will see the conversation history of a coding agent up to a key decision
point, followed by the three actions it actually took next. Write a short status
review in English that will be inserted into the history right before that point.

Requirements:
- First person, natural narrative, 2 to 5 sentences.
- Only describe what has been completed, what remains, and the verification
  evidence (tests run and their results).
- Do NOT write imperative instructions (never start a sentence with verbs like
  Run, Open, Edit, Create, Write, Add, Fix).
- If the point is a completion claim, the review must mention verification
  evidence (tests that were run and passed).
- The review must be consistent with the actions that actually happen next;
  do not mention work that only appears later.
"""
