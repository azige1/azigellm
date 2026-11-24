"""rollout 主循环：OpenAI 兼容 chat completions + 七个工具 + 推理时压缩。"""

import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from agent_rollout.tools import (
    TOOLS_SCHEMA,
    ToolResult,
    Workspace,
    execute_tool,
)
from agent_rollout.events_log import EventWriter
from agent_rollout.compact_history import CONTINUATION_PREFIX, compact_messages, find_last_read
from agent_rollout.judge import judge_run
from trajectory_tools.token_estimate import count_messages_tokens

DEFAULT_TIMEOUT_SECONDS = 3 * 3600
COMPLETION_CLAIM = "TASK COMPLETE"
CONTINUE_NUDGE = "请继续完成任务；全部完成后回复 TASK COMPLETE 并简述。"

SYSTEM_PROMPT = """你在一个工作区里完成编码任务。任务规格见用户消息：需要按规格文档
在已有代码骨架上实现整个库（骨架中函数体多为 raise NotImplementedError）。
你可以使用工具读写文件、执行命令、自行编写并运行测试来验证实现。
全部完成后，回复 {claim} 并用一两句话简述完成情况。"""


def build_system_prompt() -> str:
    """生成系统提示。"""
    return SYSTEM_PROMPT.format(claim=COMPLETION_CLAIM)


def prepare_workspace(task: dict) -> tuple[Path, Workspace]:
    """克隆任务骨架提交并建独立虚拟环境，返回 (工作区目录, 执行上下文)。

    只克隆无历史的 base 分支、不带 tag，克隆后删除 origin，
    工作区里取不到参考实现的提交对象。
    """
    import git

    from task_builder.host_environment import venv_env

    work_dir = Path(tempfile.mkdtemp(prefix="agent-rollout-"))
    repo_dir = work_dir / "repo"
    git.Repo.clone_from(
        task["local_path"],
        str(repo_dir),
        branch=task.get("base_branch", "base"),
        single_branch=True,
        no_tags=True,
    )
    repo = git.Repo(repo_dir)
    repo.git.checkout(task["base_commit"])
    repo.git.remote("remove", "origin")
    venv_dir = work_dir / ".venv"
    subprocess.run(
        ["uv", "venv", "--python", task["setup"].get("python", "3.11"), str(venv_dir)],
        check=True,
        capture_output=True,
    )
    lock = Path(task["local_path"]).parent / "requirements.lock"
    if lock.exists():
        subprocess.run(
            ["uv", "pip", "install", "-r", str(lock)],
            check=True,
            capture_output=True,
            env=venv_env(venv_dir),
        )
    # 以可编辑方式安装工作区，src 布局的包也能直接导入，Agent 不需要先处理安装问题
    subprocess.run(
        ["uv", "pip", "install", "-e", "."],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        env=venv_env(venv_dir),
    )
    # 任务目录内有参考仓库与隐藏测试，整体列为禁入路径
    forbidden = [Path(task["local_path"]).parent]
    if task.get("hidden_tests_dir"):
        forbidden.append(Path(task["hidden_tests_dir"]))
    workspace = Workspace(root=repo_dir, venv_bin=venv_dir / "bin", forbidden_dirs=forbidden)
    return repo_dir, workspace


API_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")
API_RETRIES = 4


def api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """发给接口前去掉 is_error、reasoning、g 等非标准字段。"""
    return [{k: v for k, v in m.items() if k in API_MESSAGE_KEYS} for m in messages]


def create_with_retry(client: Any, **kwargs: Any) -> Any:
    """带退避重试的 chat completions 调用，限流与网络抖动不中断整条轨迹。"""
    delay = 10.0
    for attempt in range(API_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:  # noqa: BLE001
            if attempt == API_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _write_assistant_events(writer: EventWriter, message: Any, message_id: str) -> dict:
    """把一次 assistant 回复拆成内容块事件，并返回 OpenAI 风格消息。"""
    content = message.content or ""
    if getattr(message, "reasoning_content", None):
        writer.write(
            {
                "type": "assistant",
                "message": {
                    "id": message_id,
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": message.reasoning_content}],
                },
            }
        )
    if content:
        writer.write(
            {
                "type": "assistant",
                "message": {
                    "id": message_id,
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                },
            }
        )
    openai_msg: dict[str, Any] = {"role": "assistant", "content": content}
    if getattr(message, "reasoning_content", None):
        openai_msg["reasoning"] = message.reasoning_content
    if message.tool_calls:
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
        openai_msg["tool_calls"] = tool_calls
        for call in tool_calls:
            import json

            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call["function"]["arguments"]}
            writer.write(
                {
                    "type": "assistant",
                    "message": {
                        "id": message_id,
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["function"]["name"],
                                "input": arguments,
                            }
                        ],
                    },
                }
            )
    return openai_msg


def run_rollout(
    task: dict,
    model: str,
    client: Any,
    out_dir: Path,
    max_turns: int,
    compact_at: int,
    timeout_s: int = DEFAULT_TIMEOUT_SECONDS,
    dataset_name: str = "",
    judge_timeout: int = 1800,
    keep_workspace: bool = False,
    request_extra: Optional[dict] = None,
) -> dict:
    """采集一条轨迹：主循环 + 事件日志 + 结束后判分。

    request_extra 原样作为 extra_body 传给接口，例如学生模型评测时关闭 Qwen3 thinking：
    {"chat_template_kwargs": {"enable_thinking": False}}。
    """
    run_id = uuid.uuid4().hex[:12]
    out_dir = out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(out_dir / "events.jsonl")
    workspace_dir: Optional[Path] = None
    start_time = time.time()
    num_turns = 0
    subtype = "success"
    try:
        repo_dir, workspace = prepare_workspace(task)
        workspace_dir = repo_dir
        system_prompt = build_system_prompt()
        task_text = Path(task["spec_path"]).read_text(encoding="utf-8")
        writer.write(
            {
                "type": "system",
                "subtype": "init",
                "tools": TOOLS_SCHEMA,
                "model": model,
                "session_id": run_id,
                "system_prompt": system_prompt,
                "task": task_text,
            }
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_text},
        ]
        while num_turns < max_turns:
            if time.time() - start_time > timeout_s:
                subtype = "timeout"
                break
            response = create_with_retry(
                client,
                model=model,
                messages=api_messages(messages),
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                extra_body=request_extra or None,
            )
            choice = response.choices[0]
            message = choice.message
            message_id = f"msg_{uuid.uuid4().hex[:16]}"
            openai_msg = _write_assistant_events(writer, message, message_id)
            messages.append(openai_msg)
            if not message.tool_calls:
                if COMPLETION_CLAIM in (message.content or ""):
                    subtype = "success"
                    break
                # 无工具调用且未完成：提醒继续，并写入事件流，导出时与实际上下文一致
                nudge = CONTINUE_NUDGE
                messages.append({"role": "user", "content": nudge})
                writer.write(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": [{"type": "text", "text": nudge}]},
                    }
                )
                num_turns += 1
                continue
            for call in message.tool_calls:
                result: ToolResult = execute_tool(workspace, call.function.name, call.function.arguments or "{}")
                writer.write(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call.id,
                                    "content": result.content,
                                    "is_error": result.is_error,
                                }
                            ],
                        },
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )
            num_turns += 1
            # 推理时压缩
            if count_messages_tokens(messages) > compact_at:
                last_read_path, _ = find_last_read(messages)
                last_read_content = ""
                if last_read_path:
                    candidate = workspace.root / last_read_path
                    if candidate.is_file():
                        last_read_content = candidate.read_text(encoding="utf-8", errors="replace")
                messages, summary = compact_messages(
                    system_prompt,
                    task_text,
                    messages,
                    client,
                    model,
                    last_read_content,
                    request_extra=request_extra,
                )
                writer.write({"type": "system", "subtype": "compact_boundary", "session_id": run_id})
                # 事件流写入与新上下文完全相同的摘要消息（带续接前缀）
                writer.write(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": CONTINUATION_PREFIX + "\n\n" + summary}],
                        },
                    }
                )
        else:
            subtype = "error_max_turns"
        writer.write(
            {
                "type": "result",
                "subtype": subtype,
                "num_turns": num_turns,
                "duration_s": round(time.time() - start_time, 1),
            }
        )
        # 结束后判分
        judge_run(
            repo_dir,
            task,
            run_id,
            out_dir,
            dataset_name or str(Path(task["local_path"]).parent.parent),
            timeout=judge_timeout,
        )
        kept_workspace = None
        if keep_workspace and workspace_dir is not None:
            kept_workspace = out_dir / "workspace"
            shutil.move(str(workspace_dir.parent), str(kept_workspace))
            kept_workspace = kept_workspace / "repo"
        return {
            "run_id": run_id,
            "subtype": subtype,
            "num_turns": num_turns,
            "out_dir": str(out_dir),
            "workspace": str(kept_workspace) if kept_workspace else None,
        }
    except Exception as e:  # noqa: BLE001
        writer.write(
            {
                "type": "result",
                "subtype": "timeout",
                "num_turns": num_turns,
                "duration_s": round(time.time() - start_time, 1),
                "error": str(e)[:500],
            }
        )
        return {"run_id": run_id, "subtype": "error", "num_turns": num_turns, "out_dir": str(out_dir), "error": str(e)[:500]}
    finally:
        if workspace_dir is not None and not keep_workspace:
            shutil.rmtree(workspace_dir.parent, ignore_errors=True)


__all__ = []
