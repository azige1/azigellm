"""LLM 网关接入层的单元测试。

覆盖模型注册表、模型名解析、消息格式转换、动作解析器与 pass@k 计算。
所有网络调用均 mock 掉，测试可离线运行。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_factory.llm.bedrock_access import ClaudeDeployment, get_claude_completion, available_model_aliases
from task_factory.rollout_agent import parse_agent_action
from harbor_bridge.collect_results import compute_pass_at_k


# ---------------------------------------------------------------------------
# ClaudeDeployment 注册表
# ---------------------------------------------------------------------------

class TestClaudeDeployment:
    def test_all_models_have_alias(self):
        for m in ClaudeDeployment:
            assert m.alias, f"{m.name} missing alias"

    def test_all_models_have_model_name(self):
        for m in ClaudeDeployment:
            assert m.model_name, f"{m.name} missing model_name"

    def test_model_aliases_list_matches_enum(self):
        assert set(available_model_aliases) == {m.alias for m in ClaudeDeployment}

    def test_claude_4_5_alias(self):
        assert ClaudeDeployment.CLAUDE_4_5.alias == "claude_4_5"

    def test_claude_opus_alias(self):
        assert ClaudeDeployment.CLAUDE_OPUS.alias == "claude_opus"

    def test_unique_aliases(self):
        aliases = [m.alias for m in ClaudeDeployment]
        assert len(aliases) == len(set(aliases))

    def test_unique_model_names(self):
        names = [m.model_name for m in ClaudeDeployment]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# get_claude_completion（全部 mock）
# ---------------------------------------------------------------------------

def _make_gateway_response(text: str) -> dict:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "end_turn",
    }


class TestGetClaudeCompletion:
    def _patch_session(self, text: str):
        mock_client = mock.MagicMock()
        mock_client.converse.return_value = _make_gateway_response(text)
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_client
        return mock.patch("task_factory.llm.bedrock_access.Session", return_value=mock_session)

    def test_returns_string(self):
        with self._patch_session("Hello world"):
            result = get_claude_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="claude_4_5",
            )
        assert result == "Hello world"

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Model name must be one of"):
            get_claude_completion(model="nonexistent_model")

    def test_system_messages_separated(self):
        """system 消息必须走单独参数，不能混进 messages 列表。"""
        captured = {}
        mock_client = mock.MagicMock()

        def capture_converse(**kwargs):
            captured.update(kwargs)
            return _make_gateway_response("ok")

        mock_client.converse.side_effect = capture_converse
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_client

        with mock.patch("task_factory.llm.bedrock_access.Session", return_value=mock_session):
            get_claude_completion(
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                model="claude_4_5",
            )

        assert "system" in captured
        user_roles = [m["role"] for m in captured["messages"]]
        assert "system" not in user_roles

    def test_timeout_raises(self):
        """挂起的 converse 调用应在超时后抛 RuntimeError。"""
        import time

        def _hang(**kwargs):
            time.sleep(999)

        mock_client = mock.MagicMock()
        mock_client.converse.side_effect = _hang
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_client

        with mock.patch("task_factory.llm.bedrock_access.Session", return_value=mock_session):
            with mock.patch("task_factory.llm.bedrock_access.REQUEST_TIMEOUT_SEC", 0.05):
                with pytest.raises(RuntimeError, match="timed out"):
                    get_claude_completion(
                        messages=[{"role": "user", "content": "Hi"}],
                        model="claude_4_5",
                    )


# ---------------------------------------------------------------------------
# BedrockClaudeLLM 消息格式转换
# ---------------------------------------------------------------------------

class TestBedrockClaudeLLMFormatMessages:
    @pytest.fixture
    def llm(self):
        from task_factory.llm.bedrock_llm import BedrockClaudeLLM
        with mock.patch("task_factory.llm.bedrock_llm.Session"):
            return BedrockClaudeLLM(model_name="claude_4_5")

    def test_plain_user_message(self, llm):
        sys_msgs, conv_msgs = llm._format_messages("hello", [])
        assert sys_msgs == []
        assert conv_msgs == [{"role": "user", "content": [{"text": "hello"}]}]

    def test_system_message_extracted(self, llm):
        history = [{"role": "system", "content": "Be concise."}]
        sys_msgs, conv_msgs = llm._format_messages("hi", history)
        assert sys_msgs == [{"text": "Be concise."}]
        assert all(m["role"] != "system" for m in conv_msgs)

    def test_conversation_history_preserved(self, llm):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        sys_msgs, conv_msgs = llm._format_messages("third", history)
        assert len(conv_msgs) == 3
        assert conv_msgs[0]["role"] == "user"
        assert conv_msgs[1]["role"] == "assistant"
        assert conv_msgs[2]["role"] == "user"

    def test_list_content_converted(self, llm):
        history = [{"role": "user", "content": [{"text": "block"}]}]
        _, conv_msgs = llm._format_messages("next", history)
        assert conv_msgs[0]["content"] == [{"text": "block"}]

    def test_unknown_model_raises(self):
        from task_factory.llm.bedrock_llm import BedrockClaudeLLM
        with mock.patch("task_factory.llm.bedrock_llm.Session"):
            with pytest.raises(ValueError, match="Unknown model"):
                BedrockClaudeLLM(model_name="invalid_model")

    def test_get_model_context_limit(self, llm):
        assert llm.get_model_context_limit() == 200_000

    def test_get_model_output_limit(self, llm):
        assert llm.get_model_output_limit() == 32_000


# ---------------------------------------------------------------------------
# parse_agent_action（rollout_agent）
# ---------------------------------------------------------------------------

class TestParseAgentAction:
    def test_command_extracted(self):
        result = parse_agent_action("<think>plan</think><command>ls -la</command>")
        assert result == {"type": "command", "command": "ls -la"}

    def test_done_action_tag(self):
        result = parse_agent_action("<think>done</think><action>done</action>")
        assert result["type"] == "done"

    def test_done_in_command_tag(self):
        result = parse_agent_action("<command>done</command>")
        assert result["type"] == "done"

    def test_invalid_response(self):
        result = parse_agent_action("just some text with no tags")
        assert result["type"] == "invalid"

    def test_multiline_command(self):
        result = parse_agent_action("<command>echo hello\\\nworld</command>")
        assert result["type"] == "command"
        assert "echo" in result["command"]

    def test_case_insensitive_done(self):
        result = parse_agent_action("<ACTION>DONE</ACTION>")
        assert result["type"] == "done"

    def test_last_command_wins(self):
        result = parse_agent_action("<command>first</command><command>second</command>")
        assert result["command"] == "second"


# ---------------------------------------------------------------------------
# compute_pass_at_k
# ---------------------------------------------------------------------------

class TestPassAtK:
    def test_zero_correct(self):
        result = compute_pass_at_k(n=5, c=0)
        assert all(v == 0.0 for v in result.values())

    def test_all_correct(self):
        result = compute_pass_at_k(n=5, c=5)
        assert all(v == 1.0 for v in result.values())

    def test_pass_at_1_partial(self):
        result = compute_pass_at_k(n=4, c=2)
        assert 0.0 < result[1] < 1.0

    def test_keys_range(self):
        result = compute_pass_at_k(n=3, c=1)
        assert set(result.keys()) == {1, 2, 3}

    def test_monotone_increasing(self):
        result = compute_pass_at_k(n=5, c=2)
        values = [result[k] for k in sorted(result)]
        assert values == sorted(values)
