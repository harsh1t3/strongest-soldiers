import contextlib
import functools
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from openai.types.chat import ChatCompletion

from tinytroupe.clients.openai_client import OpenAIClient, logger

# Model names Claude Code accepts besides full "claude-..." ids.
_CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku", "fable"}


class ClaudeCodeClient(OpenAIClient):
    """
    Runs LLM calls through the local Claude Code CLI (`claude -p`), so they use the Claude Code
    login (e.g., a Claude subscription) instead of an API key. ANTHROPIC_API_KEY is deliberately
    not passed on, so calls are never billed to an API key by accident.

    It reuses the OpenAI client's retries, caching, concurrency limits and cost statistics:
    only the actual model call is replaced, and its result is wrapped as an OpenAI ChatCompletion.
    Sampling parameters (temperature, top_p, penalties, max tokens) are not supported by the CLI and are ignored.
    """

    def _setup_from_config(self):
        self.client = None  # no OpenAI SDK client: calls go through the CLI

    def _is_reasoning_model(self, model):
        return False

    def _count_tokens(self, messages: list, model: str):
        return None  # tiktoken counts don't apply to Claude; actual usage comes back with each response

    def _raw_model_call(self, model, chat_api_params):
        system_prompt, user_content = _to_cli_input(chat_api_params["messages"])

        # the system prompt can exceed the Windows command line limit (~32k chars), so pass it as a file
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(system_prompt)
            system_prompt_file = f.name

        try:
            command = [
                *_claude_command(), "-p",
                "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--system-prompt-file", system_prompt_file,
                # a plain model call: no tools, sessions, user/project settings (hooks, plugins) or MCP servers
                "--tools", "",
                "--no-session-persistence",
                "--setting-sources", "",
                "--strict-mcp-config",
                "--disable-slash-commands",
            ]

            cli_model = _cli_model(model)
            if cli_model is not None:
                command += ["--model", cli_model]

            response_format = chat_api_params.get("response_format")
            json_schema = _json_schema_for(response_format)
            if json_schema is not None:
                command += ["--json-schema", json.dumps(json_schema)]

            stdin_line = json.dumps({"type": "user", "message": {"role": "user", "content": user_content}})

            logger.debug(f"Calling Claude Code CLI (model={cli_model or 'Claude Code default'}, structured={json_schema is not None}).")
            completed = subprocess.run(
                command,
                input=stdin_line + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=chat_api_params.get("timeout"),
                env={k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console pop-ups on Windows
            )
        finally:
            with contextlib.suppress(OSError):  # a leftover temp file must not hide the actual outcome
                os.remove(system_prompt_file)

        _raise_for_cli_failure(completed)
        result = _final_result_event(completed.stdout)

        if json_schema is not None and result.get("structured_output") is not None:
            content = json.dumps(result["structured_output"])
        else:
            content = result.get("result", "")

        usage = result.get("usage") or {}
        prompt_tokens = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        completion_tokens = usage.get("output_tokens") or 0

        return ChatCompletion.model_validate({
            "id": result.get("session_id") or f"claude-code-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": next(iter(result.get("modelUsage") or {}), cli_model or "claude-code"),
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })


@functools.cache  # resolved once: it's on the path of every model call
def _claude_command() -> tuple:
    """
    Returns the command that starts the Claude Code CLI. Native binaries are preferred: on Windows,
    npm's claude.cmd shim runs through cmd.exe, which re-parses arguments and can mangle the JSON schema.
    """
    from tinytroupe.clients import InvalidRequestError  # avoid circular import

    home = Path.home()
    candidates = [shutil.which("claude"), home / ".local" / "bin" / "claude.exe", home / ".local" / "bin" / "claude"]
    # binaries bundled with the Claude Code extension of VS Code-based editors (newest first)
    bundled = []
    for editor_dir in (".vscode", ".cursor", ".windsurf"):
        bundled += (home / editor_dir / "extensions").glob("anthropic.claude-code-*/resources/native-binary/claude*")
    candidates += sorted(bundled, key=lambda path: path.stat().st_mtime, reverse=True)

    for candidate in candidates:
        if candidate and Path(candidate).is_file() and Path(candidate).suffix.lower() not in (".cmd", ".bat"):
            return (str(candidate),)

    # npm install on Windows: run its JavaScript entry point directly instead of the .cmd shim
    shim, node = shutil.which("claude"), shutil.which("node")
    if shim and node:
        cli_js = Path(shim).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
        if cli_js.is_file():
            return (node, str(cli_js))

    raise InvalidRequestError(
        "Claude Code CLI ('claude') not found. Install Claude Code (https://claude.com/claude-code), "
        "log in once by running `claude`, and make sure it is on your PATH."
    )


def _raise_for_cli_failure(completed: subprocess.CompletedProcess) -> None:
    """
    Raises InvalidRequestError (not retried) for failures that retrying cannot fix, and RuntimeError
    (retried with backoff) for the rest.
    """
    from tinytroupe.clients import InvalidRequestError  # avoid circular import

    result = _final_result_event(completed.stdout)
    if result is None:
        stderr = completed.stderr.strip()
        if "unknown option" in stderr:
            raise InvalidRequestError(f"This Claude Code CLI version lacks a required option, please update it (`claude update`): {stderr[:500]}")
        raise RuntimeError(f"Claude Code CLI returned no result (exit code {completed.returncode}): {stderr[:2000]}")

    if result.get("is_error"):
        message = str(result.get("result") or result.get("subtype"))
        if result.get("api_error_status") in (401, 403) or "/login" in message:
            raise InvalidRequestError(f"Claude Code is not logged in or not authorized ({message}). Run `claude` once and log in.")
        if result.get("api_error_status") == 404:
            raise InvalidRequestError(f"Claude Code rejected the request, check MODEL in config.ini: {message}")
        raise RuntimeError(f"Claude Code CLI error: {message}")


def _cli_model(model):
    """
    Returns the model to request from Claude Code, or None to use Claude Code's default.
    Non-Claude names (e.g., OpenAI models still present in config.ini) fall back to the default.
    """
    if not model:
        return None
    if model.startswith("claude") or model.split("[")[0] in _CLAUDE_MODEL_ALIASES:
        return model
    logger.debug(f"Model '{model}' is not a Claude model, using Claude Code's default model instead.")
    return None


def _json_schema_for(response_format):
    """
    Converts an OpenAI-style response format into a JSON Schema for structured output, if it has one.
    {"type": "json_object"} has no schema: the prompts already ask for JSON, which callers then extract.
    """
    if response_format is None or isinstance(response_format, dict):
        return None
    if hasattr(response_format, "model_json_schema"):  # Pydantic model class
        return response_format.model_json_schema()
    return None


def _to_cli_input(messages: list):
    """
    The CLI takes a system prompt plus one user turn, so system messages become the system prompt and
    the rest of the conversation is laid out as a transcript inside that one turn.
    Returns (system_prompt, user content blocks).
    """
    system_parts = []
    conversation = []
    for message in messages:
        if message.get("role") == "system":
            system_parts.append(_text_of(message.get("content")))
        else:
            conversation.append(message)

    label_roles = len(conversation) > 1
    blocks = []
    for message in conversation:
        if label_roles:
            blocks.append({"type": "text", "text": f"[{message.get('role', 'user')}]"})
        blocks.extend(_content_blocks(message.get("content")))

    if not blocks:
        blocks.append({"type": "text", "text": "Continue."})

    system_prompt = "\n\n".join(p for p in system_parts if p) or "You are a helpful assistant."
    return system_prompt, blocks


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return "" if content is None else str(content)


def _content_blocks(content) -> list:
    """Converts OpenAI message content (a string or a list of text/image_url parts) to Anthropic content blocks."""
    if not isinstance(content, list):
        text = _text_of(content)
        return [{"type": "text", "text": text}] if text.strip() else []

    blocks = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text", "").strip():
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                # data:<media type>;base64,<data>
                header, _, data = url.partition(",")
                media_type = header[len("data:"):].split(";")[0]
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _final_result_event(stdout: str):
    """Returns the final "result" event of the CLI's stream-json output."""
    result = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    return result
