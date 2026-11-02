"""九段摘要提示词：推理时压缩与压缩恢复样本共用。

每段 1 至 5 句；输出中不得出现 compaction、summary 等自指词。
"""

NINE_SECTION_SUMMARY_PROMPT = """把下面的对话历史压缩成九段回顾，每段 1 到 5 句，只陈述事实，不写操作建议。
各段标题与顺序固定：

1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Errors and Fixes
5. Problem Solving
6. All User Messages
7. Pending Tasks
8. Current Work
9. Optional Next Step

要求：使用英文；不要出现 compaction、compact、summary、summarize、
context limit、context window 等词；保留关键文件路径与错误信息。
"""


def nine_section_prompt() -> str:
    """返回九段摘要的系统提示词。"""
    return NINE_SECTION_SUMMARY_PROMPT
