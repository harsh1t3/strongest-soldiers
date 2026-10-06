"""
Offline tests for the Claude Code CLI client (the CLI itself is replaced by a fake).
"""
import json
import subprocess

import pytest
from pydantic import BaseModel

from tinytroupe.clients import claude_code_client, InvalidRequestError
from tinytroupe.clients.claude_code_client import ClaudeCodeClient, _cli_model, _to_cli_input


class _Answer(BaseModel):
    value: int


def test_messages_become_system_prompt_and_transcript():
    system, blocks = _to_cli_input([
        {"role": "system", "content": "Be Oscar."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": [
            {"type": "text", "text": "Look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ])
    assert system == "Be Oscar."
    assert [b.get("text") for b in blocks if b["type"] == "text"] == ["[user]", "Hi", "[assistant]", "Hello", "[user]", "Look"]
    assert blocks[-1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}


def test_non_claude_models_fall_back_to_the_cli_default():
    assert _cli_model("sonnet") == "sonnet"
    assert _cli_model("claude-haiku-4-5") == "claude-haiku-4-5"
    assert _cli_model("gpt-5-mini") is None


def test_raw_call_wraps_cli_result_as_chat_completion(monkeypatch):
    seen = {}

    def fake_run(command, input, **kwargs):
        seen["command"] = command
        seen["input"] = json.loads(input)
        seen["env"] = kwargs["env"]
        result = {"type": "result", "is_error": False, "result": '{"value": 7}', "structured_output": {"value": 7},
                  "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3}}
        return subprocess.CompletedProcess(command, 0, stdout='{"type": "system"}\n' + json.dumps(result) + "\n", stderr="")

    monkeypatch.setattr(claude_code_client, "_claude_command", lambda: ("claude",))
    monkeypatch.setattr(claude_code_client.subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")

    response = ClaudeCodeClient()._raw_model_call(
        "haiku", {"messages": [{"role": "user", "content": "Pick a number"}], "response_format": _Answer, "timeout": 5})

    assert json.loads(response.choices[0].message.content) == {"value": 7}
    assert response.usage.prompt_tokens == 15 and response.usage.completion_tokens == 3
    assert seen["command"][seen["command"].index("--model") + 1] == "haiku"
    assert json.loads(seen["command"][seen["command"].index("--json-schema") + 1])["required"] == ["value"]
    assert seen["input"]["message"]["content"] == [{"type": "text", "text": "Pick a number"}]
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def _completed(result=None, stderr="", returncode=0):
    stdout = json.dumps({"type": "result", **result}) + "\n" if result is not None else ""
    return subprocess.CompletedProcess(["claude"], returncode, stdout=stdout, stderr=stderr)


def test_unfixable_cli_failures_are_not_retried():
    with pytest.raises(InvalidRequestError):
        claude_code_client._raise_for_cli_failure(_completed({"is_error": True, "result": "Not logged in · Please run /login"}))
    with pytest.raises(InvalidRequestError):
        claude_code_client._raise_for_cli_failure(_completed(stderr="error: unknown option '--json-schema'", returncode=1))
    # transient failures are retried by the caller
    with pytest.raises(RuntimeError):
        claude_code_client._raise_for_cli_failure(_completed({"is_error": True, "result": "Overloaded", "api_error_status": 529}))
    claude_code_client._raise_for_cli_failure(_completed({"is_error": False, "result": "ok"}))


def test_npm_cmd_shim_is_bypassed(monkeypatch, tmp_path):
    shim = tmp_path / "claude.cmd"
    shim.write_text("@echo off")
    cli_js = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
    cli_js.parent.mkdir(parents=True)
    cli_js.write_text("")
    monkeypatch.setattr(claude_code_client.shutil, "which", lambda name: {"claude": str(shim), "node": "node.exe"}[name])
    monkeypatch.setattr(claude_code_client.Path, "home", lambda: tmp_path / "empty-home")

    claude_code_client._claude_command.cache_clear()
    try:
        assert claude_code_client._claude_command() == ("node.exe", str(cli_js))
    finally:
        claude_code_client._claude_command.cache_clear()
