"""token 估算与盲改等行为指标共用的轨迹消息模型。

token 估算口径统一为「字符数 / 3」，粗估只用于压缩阈值、注入预算与
切点判断；打包截断前会用 tokenizer 实计（见 training/build_dataset.py）。
"""

CHARS_PER_TOKEN = 3


def count_text_tokens(text: str) -> int:
    """估算一段文本的 token 数。"""
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def count_message_tokens(message: dict) -> int:
    """估算一条 OpenAI 风格消息的 token 数（含工具调用）。"""
    total = count_text_tokens(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        total += count_text_tokens(str(call.get("function", {}).get("arguments", "")))
    return total


def count_messages_tokens(messages: list[dict]) -> int:
    """估算一串消息的总 token 数。"""
    return sum(count_message_tokens(m) for m in messages)


__all__ = []
