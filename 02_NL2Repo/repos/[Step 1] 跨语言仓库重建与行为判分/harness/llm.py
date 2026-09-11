"""LLM 接入与两种 agent 对话循环。

- chat_completion：一次 OpenAI 兼容的 chat 调用。
- run_tool_agent：函数调用（tool_calls）协议的多轮循环，模型通过 execute_shell 工具下命令。
- run_text_agent：纯文本协议的多轮循环，模型用 <tool_call>...</tool_call> 包裹命令，
  用于在 Docker 容器里执行的版本（不依赖模型的函数调用能力）。

两种循环都返回 token 用量字典，并可选返回完整消息轨迹。
"""

import json
import re

from . import config

SHELL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": "Execute a Linux shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full shell command to run."}
                },
                "required": ["command"],
            },
        },
    }
]

TOOL_SYSTEM_PROMPT = (
    "You are an AI expert with Linux terminal access. "
    "Use your reasoning ability before calling any tools."
)

TEXT_SYSTEM_PROMPT = """You are an AI expert with Linux terminal access inside a Docker container. You can execute commands inside the container to complete tasks.

When you need to execute a command, use this exact format:
<tool_call>
command content here
</tool_call>

Important rules:
- Execute only one command at a time
- Wait for command results before deciding next steps
- When the task is complete, give a final summary WITHOUT any <tool_call> tags
- All operations are performed in an isolated container environment
"""


def _make_client():
    """按环境变量构造 OpenAI 兼容客户端，未配置时给出明确报错。"""
    from openai import OpenAI

    if not config.LLM_API_KEY:
        raise ValueError("请先设置环境变量 LLM_API_KEY（以及可选的 LLM_BASE_URL）")
    return OpenAI(base_url=config.LLM_BASE_URL or None, api_key=config.LLM_API_KEY)


def chat_completion(messages, model, tools=None):
    """发起一次 chat 调用，成功返回 dict，失败打印原因并返回 None。"""
    client = _make_client()
    params = {"model": model, "messages": messages}
    if tools:
        params["tools"] = tools
    try:
        response = client.chat.completions.create(**params)
        return response.model_dump()
    except Exception as exc:
        print(f"[ERROR] 模型调用失败: {exc}")
        return None


def run_tool_agent(user_query, model_name, execute, after_command=None,
                   max_total_tokens=2_560_000, return_messages=False):
    """函数调用协议的 agent 循环。

    参数:
        user_query: 任务提示词。
        model_name: 模型名。
        execute: 回调，接收 shell 命令字符串，返回观测字符串。
        after_command: 可选回调，每执行完一条命令后调用，返回额外提示或 None
            （用于在循环外做产物检查，例如剥离 Cargo.toml 里的外部依赖）。
        max_total_tokens: 累计 token 上限，超过即终止。
        return_messages: 是否在结果里带回完整消息轨迹。
    """
    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    total_input = 0
    total_output = 0

    while True:
        response = chat_completion(messages, model=model_name, tools=SHELL_TOOL)
        if not response or "choices" not in response:
            print("[ERROR] 模型调用失败或返回格式异常")
            break

        usage = response.get("usage") or {}
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)

        current_total = total_input + total_output
        if current_total >= max_total_tokens:
            print(f"\n[STOP] 达到 token 上限: {current_total}/{max_total_tokens}")
            result = {
                "error": f"token limit reached ({max_total_tokens})",
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": current_total,
            }
            if return_messages:
                result["messages"] = messages
            return result

        msg = response["choices"][0]["message"]

        if msg.get("reasoning_details"):
            print(f"\n[reasoning]\n{msg['reasoning_details']}")

        msg_to_append = {"role": "assistant", "content": msg.get("content", "")}
        if "reasoning_details" in msg:
            msg_to_append["reasoning_details"] = msg["reasoning_details"]
        if msg.get("tool_calls"):
            msg_to_append["tool_calls"] = msg["tool_calls"]
        messages.append(msg_to_append)

        if not msg.get("tool_calls"):
            print(f"\n[final answer]\n{msg.get('content', '')}")
            break

        for tool_call in msg["tool_calls"]:
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as exc:
                observation = (
                    f"Invalid tool arguments: {exc}\n"
                    f"Raw: {tool_call['function']['arguments']}\n"
                    "Please fix the argument format and retry."
                )
            else:
                observation = execute(args["command"])
                if after_command:
                    extra = after_command()
                    if extra:
                        observation = f"{observation}\n\n{extra}"
            print(f"[tool result]\n{observation[:1000]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": "execute_shell",
                "content": observation,
            })

    result = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }
    if return_messages:
        result["messages"] = messages
    return result


def parse_text_tool_call(content):
    """从模型输出里解析 <tool_call> 包裹的命令，没有则返回 None。"""
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_text_agent(user_query, model_name, execute, max_turns=12, return_messages=False):
    """纯文本协议的 agent 循环（容器版默认使用）。

    参数:
        user_query: 任务提示词。
        model_name: 模型名。
        execute: 回调，接收 shell 命令字符串，返回观测字符串。
        max_turns: 最大对话轮数，达到后强制停止。
        return_messages: 是否在结果里带回完整消息轨迹。
    """
    messages = [
        {"role": "system", "content": TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    total_input = 0
    total_output = 0

    for _ in range(max_turns):
        clean_messages = [
            {"role": m["role"], "content": str(m.get("content") or "")} for m in messages
        ]
        response = chat_completion(clean_messages, model=model_name)
        if not response or "choices" not in response:
            print("[ERROR] 模型调用失败或返回格式异常")
            break

        usage = response.get("usage") or {}
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)

        msg = response["choices"][0]["message"]
        content = msg.get("content", "") or ""

        if msg.get("reasoning_details"):
            print(f"\n[reasoning]\n{msg['reasoning_details']}")

        messages.append({"role": "assistant", "content": content})

        command = parse_text_tool_call(content)
        if not command:
            print(f"\n[final answer]\n{content}")
            break

        observation = execute(command)
        print(f"[tool result]\n{observation[:1000]}")
        messages.append({"role": "user", "content": f"<tool_result>\n{observation}\n</tool_result>"})
    else:
        print(f"\n[STOP] 达到最大轮数 max_turns={max_turns}")

    result = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }
    if return_messages:
        result["messages"] = messages
    return result
