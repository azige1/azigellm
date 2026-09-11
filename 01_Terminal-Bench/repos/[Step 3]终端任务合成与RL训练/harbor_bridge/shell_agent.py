"""接入 Harbor 基准框架的单命令终端 agent。

复用解题管线的提示模板（``AGENT_SYSTEM_PROMPT`` / ``AGENT_USER_TEMPLATE``），
每轮让模型输出一条 shell 命令或 done，然后通过 ``environment.exec()``
直接在任务容器里执行。
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict, Field

from transformers import AutoTokenizer

# 终端输出清洗：ANSI 转义与 shell 启动噪声
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

SHELL_NOISE_PATTERNS = [
    re.compile(r"^bash: cannot set terminal process group.*$", re.MULTILINE),
    re.compile(r"^bash: no job control in this shell.*$", re.MULTILINE),
    re.compile(r"^mesg: ttyname failed:.*$", re.MULTILINE),
    re.compile(r"^stdin: is not a tty.*$", re.MULTILINE),
    re.compile(r"^the input device is not a TTY.*$", re.MULTILINE),
    re.compile(r"^warning: TERM environment variable not set.*$", re.MULTILINE | re.IGNORECASE),
]

# 允许从仓库根目录导入 task_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_factory.rollout_agent import AGENT_SYSTEM_PROMPT, AGENT_USER_TEMPLATE, parse_agent_action
from task_factory import batch_chat_completions

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


MAX_OUTPUT_LENGTH = 50000


def clean_terminal_output(output: str) -> str:
    """清洗命令输出：去 ANSI 转义、回车符与 shell 启动噪声。"""
    cleaned = ANSI_RE.sub("", output)

    cleaned = cleaned.replace("\r", "")

    for pattern in SHELL_NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


class AgentAction(BaseModel):
    """模型每轮必须返回的严格 JSON 结构。"""
    type: Literal["command", "done", "invalid"] = Field(
        description="Choose 'command' to run a shell command, or 'done' to finish."
    )
    command: Optional[str] = Field(
        default=None,
        description="Required when type == 'command'. Exact shell command to run.",
    )
    timeout_sec: Optional[float] = Field(
        default=180.0,
        description="Optional per-command timeout (seconds) when blocking on output.",
    )

    model_config = ConfigDict(extra="forbid")


class TerminalShellAgent(BaseAgent):
    """面向 Harbor 基准的单命令终端 agent。

    按固定提示模板驱动 LLM，逐轮解析 ``<command>`` / ``<action>done</action>``，
    并通过 ``environment.exec()`` 直接执行命令。
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *args,
        temperature: float = 0.6,
        max_episodes: int = 64,
        max_time_sec: float = 600,
        max_completion_tokens: int = 2048,
        agent_version: str = "1.0.0",
        tokenizer_model: str | None = None,
        **kwargs,
    ):
        """
        参数:
            logs_dir: 日志目录。
            model_name: 补全所用模型（可以是推理端点 URL）。
            temperature: 采样温度。
            max_episodes: 最大交互轮数。
            max_time_sec: 单任务墙钟时间上限。
            max_completion_tokens: 单次补全的最大 token 数。
            agent_version: agent 版本号。
            tokenizer_model: 分词器的 HF 模型名/路径，缺省回退到 model_name。
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self._model_name = model_name or "Qwen/Qwen2.5-3B-Instruct"
        self._tokenizer_model = tokenizer_model or self._model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._max_episodes = max_episodes
        self._max_time = max_time_sec
        self._agent_version = agent_version
        self._chat: list[dict[str, str]] = []
        self._original_first_user_content: str = ""  # 保留首条 user 原文，避免历史摘要反复叠加
        self.start_time = time.time()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._tokenizer = None
        self._tokenizer_loaded = False

    @staticmethod
    def name() -> str:
        """agent 名称。"""
        return "terminal-shell"

    def version(self) -> str | None:
        """agent 版本。"""
        return self._agent_version

    async def setup(self, environment: BaseEnvironment) -> None:
        """任务开始前的环境准备：重置状态并复位终端字符集。"""
        self._chat = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # \ec 是 RIS（Reset to Initial State）转义序列，修复终端字符集错乱
        await environment.exec(command="echo -e '\\ec'", timeout_sec=5)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """在环境中执行任务，并把 token 用量写入 context。"""
        initial_user = self._render_initial_user(instruction)

        self._chat = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_user},
        ]
        self._original_first_user_content = initial_user

        n_episodes = 0
        start = time.perf_counter()
        try:
            for episode in range(1, self._max_episodes + 1):
                n_episodes = episode

                decision, raw_response = await self._ask_for_decision()

                self._chat.append({"role": "assistant", "content": raw_response})

                if decision["type"] == "done":
                    print(f"[Episode {episode}] Agent finished (done)")
                    break

                elif decision["type"] == "command":
                    cmd = decision["command"] or ""
                    timeout = decision.get("timeout_sec", 180.0) or 180.0
                    print(f"[Episode {episode}] Sending command (timeout={timeout}s): {cmd}")

                    try:
                        result = await environment.exec(command=cmd, timeout_sec=int(300))
                        exit_code = result.return_code
                        output = (result.stdout or "") + (result.stderr or "")

                        output = clean_terminal_output(output)
                        output, truncated_msg = self._truncate_output(output)

                        print(f"[Episode {episode}] Output (exit={exit_code}): {output[:500]}{'...' if len(output) > 500 else ''}")

                        if exit_code == 0:
                            result_back = f"Command executed successfully. Output: {output}{truncated_msg}\n\n(exit_code={exit_code})"
                        else:
                            result_back = f"Command failed. Output: {output}{truncated_msg}\n\n(exit_code={exit_code})"

                    except Exception as e:
                        print(f"[Episode {episode}] Command failed with error: {e}")
                        result_back = f"Command failed with error: {str(e)}\n\n(exit_code=1)"

                    self._chat.append({"role": "user", "content": result_back})

                else:
                    # 解析失败，提醒模型按协议回复
                    result_back = (
                        "Could not parse a single <command>...</command> or <action>done</action>. "
                        "Please respond with exactly one of those."
                    )
                    self._chat.append({"role": "user", "content": result_back})

                # 墙钟超时保护
                if (time.perf_counter() - start) > self._max_time:
                    break
        finally:
            context.n_input_tokens = self._total_input_tokens
            context.n_output_tokens = self._total_output_tokens
            context.cost_usd = None
            context.metadata = {
                "n_episodes": n_episodes,
                "all_messages": self._chat,
            }

    def _truncate_output(self, output: str) -> tuple[str, str]:
        """截断超长输出，返回 ``(截断后的输出, 追加提示)``。"""
        truncated_msg = ""
        original_len = len(output)
        if original_len > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH]
            truncated_msg = f"\n[Output truncated: showing first {MAX_OUTPUT_LENGTH} of {original_len} characters]"
        return output, truncated_msg

    def _get_tokenizer(self):
        """惰性加载并缓存分词器。"""
        if self._tokenizer_loaded:
            return self._tokenizer

        self._tokenizer_loaded = True


        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_model)
        print(f"[Info] Loaded tokenizer for {self._tokenizer_model}")


        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """用模型分词器统计文本 token 数。"""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        """按聊天模板统计整段对话的 token 数。"""
        tokenizer = self._get_tokenizer()
        total = 0
        chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(chat)

    def _render_initial_user(self, instruction: str) -> str:
        """渲染首条 user 消息：题面 + 非交互限制说明。"""
        restrictions = (
            "\n\nRESTRICTIONS:\n"
            "- You cannot use sudo or any commands requiring root privileges.\n"
            "- You cannot use interactive tools like vim, nano, etc.\n"
            "- When running commands that prompt for input (yes/no, etc.), use non-interactive flags (e.g., `-y`, `--yes`, `--non-interactive`) when available, or pipe the inputs (e.g., `echo -e 'yes\\nno' | ./script.sh` or `yes | ./script.sh`) since you cannot interact with running processes.\n"
        )
        instruction_with_restrictions = instruction + restrictions

        try:
            return AGENT_USER_TEMPLATE.format(task_description=instruction_with_restrictions)
        except Exception as e:
            raise ValueError(
                "task_factory.rollout_agent.AGENT_USER_TEMPLATE must include '{task_description}'."
            ) from e

    def _build_command_history_summary(self) -> str:
        """把对话历史压缩为「命令 + 输出」摘要，供上下文截断后重试使用。"""
        history_parts = []
        # 跳过 system（下标 0）与首条 user（下标 1）
        # 对话结构：system, user, assistant, user(输出), assistant, user(输出), ...
        i = 2
        cmd_num = 1
        while i < len(self._chat):
            msg = self._chat[i]
            if msg["role"] == "assistant":
                action = parse_agent_action(msg["content"])
                if action["type"] == "command" and action.get("command"):
                    cmd = action["command"]
                    output = ""
                    if i + 1 < len(self._chat) and self._chat[i + 1]["role"] == "user":
                        output = self._chat[i + 1]["content"]
                        if len(output) > 500:
                            output = output[:500] + "... [truncated]"
                    history_parts.append(f"Command {cmd_num}: {cmd}\nOutput: {output}")
                    cmd_num += 1
            i += 1

        if not history_parts:
            return ""

        return "Here is a summary of commands and outputs from your previous attempts:\n\n" + "\n\n---\n\n".join(history_parts) + "\n\nPlease continue from where you left off."

    async def _ask_for_decision(self, retry_on_error: bool = True) -> tuple[dict, str]:
        """向模型请求下一步动作，返回 ``(解析后的动作, 原始输出)``。

        补全失败（返回 None）时压缩历史后重试一次。
        """
        # 同步的批量补全放到执行器里，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            lambda: batch_chat_completions(
                [self._chat],
                model=self._model_name,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
                num_completions=1,
                max_concurrency=1,
            )
        )

        if responses[0] is None:
            print("[Warning] Chat completion returned None, likely context too long or API error")

            if retry_on_error and len(self._chat) > 2:
                history_summary = self._build_command_history_summary()

                # 保留 system，并用首条 user 原文重建上下文，
                # 避免历史摘要在多次重试间叠加
                system_msg = self._chat[0]
                first_user_content = self._original_first_user_content

                if history_summary:
                    first_user_content = first_user_content + "\n\n" + history_summary

                self._chat = [system_msg, {"role": "user", "content": first_user_content}]

                print("[Info] Truncated chat and added command history summary, retrying...")

                # 只允许重试一次，防止死循环
                return await self._ask_for_decision(retry_on_error=False)
            else:
                return {"type": "invalid", "command": None}, "Error: API returned None response"

        raw_str = responses[0].choices[0].message.content

        input_tokens = self._count_chat_tokens(self._chat)
        output_tokens = self._count_tokens(raw_str)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        print(f"[Tokens] input={input_tokens}, output={output_tokens}, total_input={self._total_input_tokens}, total_output={self._total_output_tokens}")

        action = parse_agent_action(raw_str)
        return action, raw_str
