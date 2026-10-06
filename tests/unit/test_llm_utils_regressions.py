"""
Offline regression tests for the @llm decorator and the LLM clients' cache/config handling.
No model is called: the LLM call is replaced by fakes.
"""
import datetime
import json
import os
import threading
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from tinytroupe import config_manager
from tinytroupe.clients.ollama_client import OllamaClient
from tinytroupe.clients.openai_client import LLMCacheBase, OpenAIClient
from tinytroupe.utils.llm import LLMChat, llm


def _rendered_parameters(call):
    """Runs `call` with the LLM call faked, returning the input parameters rendered in the user prompt."""
    prompts = []

    def fake_call(self, *args, **kwargs):
        prompts.append(self.user_prompt)
        return "ok"

    with patch.object(LLMChat, "call", fake_call):
        call()

    return json.loads(prompts[0].split("## Input parameters\n")[1].split("\n\n")[0])


def test_llm_decorator_names_positional_arguments_and_fills_defaults():
    @llm()
    def extract(query: str, text: str, context: str = None) -> str:
        """Extracts information."""

    assert _rendered_parameters(lambda: extract("job?", "Maria is a nurse.")) == {
        "query": "job?",
        "text": "Maria is a nurse.",
        "context": None,
    }


def test_llm_decorator_drops_self_and_renders_non_json_arguments_as_text():
    class Memory:
        @llm()
        def consolidate(self, memories: list, timestamp, context: str, persona) -> dict:
            """Consolidates memories."""

    class Persona:
        def __str__(self):
            return "a cat lover"

    when = datetime.datetime(2024, 1, 2, 3, 4, 5)
    params = _rendered_parameters(lambda: Memory().consolidate([{"a": 1}], when, "ctx", persona=Persona()))

    assert params == {"memories": [{"a": 1}], "timestamp": str(when), "context": "ctx", "persona": "a cat lover"}


def test_cache_save_is_atomic_when_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # relative cache file name, as in config.ini
    cache = LLMCacheBase()
    cache.cache_file_name = "api_cache.json"
    cache.api_cache = {"old": "entry"}
    cache._save_cache()

    cache.api_cache["new"] = "entry"

    def crash_mid_write(obj, f, **kwargs):
        f.write('{"old": "entry", "ne')
        raise KeyboardInterrupt  # e.g., the user stops the simulation

    with patch("tinytroupe.clients.openai_client.json.dump", crash_mid_write):
        try:
            cache._save_cache()
        except KeyboardInterrupt:
            pass

    assert json.loads((tmp_path / "api_cache.json").read_text(encoding="utf-8")) == {"old": "entry"}
    assert os.listdir(tmp_path) == ["api_cache.json"]  # no leftover temp file


def test_ollama_cache_is_safe_under_concurrent_calls(tmp_path):
    cache_file = str(tmp_path / "ollama_cache.json")
    client = OllamaClient(cache_api_calls=True, cache_file_name=cache_file)
    # a big cache makes each save slow, so other threads add entries while one is being written
    client.api_cache.update({f"existing {i}": {"content": "x" * 50} for i in range(1000)})

    save_errors = []
    original_save = client._save_cache

    def recording_save():
        try:
            original_save()
        except Exception as e:
            save_errors.append(e)
            raise

    client._save_cache = recording_save
    response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    results = []

    def worker(thread_id):
        for j in range(10):
            messages = [{"role": "user", "content": f"message {thread_id}-{j}"}]
            results.append(client.send_message(messages, model="m", waiting_time=0, max_attempts=1))

    with patch.object(OllamaClient, "_make_request", return_value=response):
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert save_errors == []
    assert results == [{"role": "assistant", "content": "hi"}] * 40
    with open(cache_file, encoding="utf-8") as f:
        assert len(json.load(f)) == 1000 + 4 * 10


def test_openai_client_uses_top_p_from_config(monkeypatch, tmp_path):
    monkeypatch.setitem(config_manager._config, "top_p", 0.3)
    client = OpenAIClient(cache_api_calls=False, cache_file_name=str(tmp_path / "cache.json"))
    sent_params = []

    def fake_model_call(model, chat_api_params):
        sent_params.append(chat_api_params)
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        })

    with patch.object(client, "_setup_from_config"), patch.object(client, "_raw_model_call", fake_model_call):
        client.send_message([{"role": "user", "content": "hello"}], model="m", max_attempts=1, waiting_time=0)

    assert sent_params[0]["top_p"] == 0.3
