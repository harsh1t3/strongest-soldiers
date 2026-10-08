"""
Tests for the LLM API caching mechanism in OpenAIClient and LLMCacheBase.

These tests verify that the cache save/load functionality works correctly,
ensuring API responses can be persisted to disk as JSON and reloaded.
"""
import json
import os
import tempfile
import pytest

from tinytroupe.clients.openai_client import LLMCacheBase, OpenAIClient
from tinytroupe.clients.ollama_client import OllamaClient


class TestLLMCacheBase:
    """Tests for the LLMCacheBase cache functionality."""

    @pytest.fixture
    def temp_cache_file(self):
        """Creates a temporary file path for cache testing (file is not created)."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.mark.core
    def test_save_and_load_cache(self, temp_cache_file):
        """Test that _save_cache and _load_cache work with JSON."""
        cache = LLMCacheBase()
        cache.cache_file_name = temp_cache_file
        cache.api_cache = {"key1": {"response": "value1"}}
        cache._save_cache()

        assert os.path.exists(temp_cache_file), "Cache file should exist after saving"
        with open(temp_cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == {"key1": {"response": "value1"}}

    @pytest.mark.core
    def test_load_cache_nonexistent_file(self, temp_cache_file):
        """Test that _load_cache returns empty dict for nonexistent file."""
        cache = LLMCacheBase()
        cache.cache_file_name = temp_cache_file
        result = cache._load_cache()
        assert result == {}

    @pytest.mark.core
    def test_load_cache_corrupted_file(self, temp_cache_file):
        """Test that _load_cache gracefully handles corrupted JSON."""
        with open(temp_cache_file, "w") as f:
            f.write("this is not valid json {{{")
        cache = LLMCacheBase()
        cache.cache_file_name = temp_cache_file
        result = cache._load_cache()
        assert result == {}

    @pytest.mark.core
    def test_load_cache_empty_file(self, temp_cache_file):
        """Test that _load_cache gracefully handles an empty file."""
        with open(temp_cache_file, "w") as f:
            pass
        cache = LLMCacheBase()
        cache.cache_file_name = temp_cache_file
        result = cache._load_cache()
        assert result == {}

    @pytest.mark.core
    def test_set_api_cache_enables_and_disables(self, temp_cache_file):
        """Test set_api_cache enables/disables caching."""
        cache = LLMCacheBase()
        cache.set_api_cache(True, temp_cache_file)
        assert cache.cache_api_calls is True
        assert cache.cache_file_name == temp_cache_file
        assert hasattr(cache, 'api_cache')

        cache.set_api_cache(False, temp_cache_file)
        assert cache.cache_api_calls is False

    @pytest.mark.core
    def test_cache_roundtrip_with_unicode(self, temp_cache_file):
        """Test that cache handles Unicode and special characters via JSON."""
        cache = LLMCacheBase()
        cache.cache_file_name = temp_cache_file
        cache.api_cache = {
            "unicode_key": "Hello 世界 🌍 مرحبا",
            "newlines": "line1\nline2\r\nline3",
            "quotes": 'single \' and "double" quotes',
        }
        cache._save_cache()

        cache2 = LLMCacheBase()
        cache2.cache_file_name = temp_cache_file
        loaded = cache2._load_cache()
        assert loaded["unicode_key"] == "Hello 世界 🌍 مرحبا"
        assert loaded["newlines"] == "line1\nline2\r\nline3"


class TestOpenAIClientCache:
    """Tests for OpenAIClient cache functionality."""

    @pytest.fixture
    def temp_cache_file(self):
        """Creates a temporary file path for cache testing (file is not created)."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.mark.core
    def test_save_cache_creates_file(self, temp_cache_file):
        """Test that _save_cache successfully creates a JSON file."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        # Add some data to the cache
        client.api_cache["test_key"] = {"response": "test_value"}
        
        # Save the cache
        client._save_cache()
        
        # Verify file was created and contains correct data
        assert os.path.exists(temp_cache_file), "Cache file should exist after saving"
        
        with open(temp_cache_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert "test_key" in loaded_data
        assert loaded_data["test_key"] == {"response": "test_value"}

    @pytest.mark.core
    def test_load_cache_reads_file(self, temp_cache_file):
        """Test that _load_cache successfully reads a JSON file."""
        # First, create a cache file with known data
        test_data = {"existing_key": {"cached": "response"}}
        with open(temp_cache_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        
        # Create client and verify it loads the cache
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        assert "existing_key" in client.api_cache
        assert client.api_cache["existing_key"] == {"cached": "response"}

    def test_load_cache_returns_empty_dict_for_nonexistent_file(self, temp_cache_file):
        """Test that _load_cache returns empty dict when file doesn't exist."""
        # Remove the file if it exists
        if os.path.exists(temp_cache_file):
            os.remove(temp_cache_file)
        
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        assert client.api_cache == {}

    @pytest.mark.core
    def test_cache_roundtrip(self, temp_cache_file):
        """Test that data survives a save/load roundtrip."""
        # Create client and add complex data
        client1 = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        complex_data = {
            "key1": {"nested": {"deeply": "value"}},
            "key2": [1, 2, 3, "mixed", {"types": True}],
            "key3": "simple string",
        }
        
        for key, value in complex_data.items():
            client1.api_cache[key] = value
        
        client1._save_cache()
        
        # Create new client that loads the cache
        client2 = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        # Verify all data was preserved
        for key, value in complex_data.items():
            assert key in client2.api_cache, f"Key '{key}' should be in loaded cache"
            assert client2.api_cache[key] == value, f"Value for '{key}' should match"

    def test_set_api_cache_enables_caching(self, temp_cache_file):
        """Test that set_api_cache properly enables caching."""
        client = OpenAIClient(cache_api_calls=False, cache_file_name=temp_cache_file)
        
        assert client.cache_api_calls is False
        
        # Enable caching
        client.set_api_cache(True, temp_cache_file)
        
        assert client.cache_api_calls is True
        assert client.cache_file_name == temp_cache_file
        assert hasattr(client, 'api_cache')

    def test_set_api_cache_disables_caching(self, temp_cache_file):
        """Test that set_api_cache properly disables caching."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        assert client.cache_api_calls is True
        
        # Disable caching
        client.set_api_cache(False, temp_cache_file)
        
        assert client.cache_api_calls is False

    def test_cache_with_special_characters(self, temp_cache_file):
        """Test that cache handles data with special characters."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        # Data with unicode and special characters
        special_data = {
            "unicode_key": "Hello 世界 🌍 مرحبا",
            "newlines": "line1\nline2\r\nline3",
            "quotes": 'single \' and "double" quotes',
        }
        
        for key, value in special_data.items():
            client.api_cache[key] = value
        
        client._save_cache()
        
        # Load in new client
        client2 = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        for key, value in special_data.items():
            assert client2.api_cache[key] == value

    def test_get_cached_response_returns_none_when_disabled(self, temp_cache_file):
        """Test that _get_cached_response returns None when caching is disabled."""
        client = OpenAIClient(cache_api_calls=False, cache_file_name=temp_cache_file)
        
        result = client._get_cached_response("any_key")
        
        assert result is None

    @pytest.mark.core
    def test_get_cached_response_returns_cached_value(self, temp_cache_file):
        """Test that _get_cached_response returns cached value when available."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        # Create a valid ChatCompletion-like dict structure for the cache
        cached_response = {
            "id": "chatcmpl-test123",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "cached_value",
                        "role": "assistant"
                    }
                }
            ],
            "created": 1234567890,
            "model": "gpt-4",
            "object": "chat.completion"
        }
        
        # Add properly formatted data to cache
        client.api_cache["test_key"] = cached_response
        
        result = client._get_cached_response("test_key")
        
        assert result is not None, "Should return a ChatCompletion object"
        assert result.choices[0].message.content == "cached_value"

    def test_get_cached_response_returns_none_for_missing_key(self, temp_cache_file):
        """Test that _get_cached_response returns None for missing key."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        result = client._get_cached_response("nonexistent_key")
        
        assert result is None

    def test_load_cache_handles_empty_file(self, temp_cache_file):
        """Test that _load_cache gracefully handles an empty file."""
        # Create an empty file
        with open(temp_cache_file, "w") as f:
            pass  # creates empty file
        
        # Should not raise, should return empty dict
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        assert client.api_cache == {}

    @pytest.mark.core
    def test_load_cache_handles_corrupted_file(self, temp_cache_file):
        """Test that _load_cache gracefully handles a corrupted file."""
        # Create a file with invalid JSON data
        with open(temp_cache_file, "w") as f:
            f.write("this is not valid json data")
        
        # Should not raise, should return empty dict
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        
        assert client.api_cache == {}

    @pytest.mark.core
    def test_openai_client_inherits_from_cache_base(self):
        """Test that OpenAIClient inherits from LLMCacheBase."""
        assert issubclass(OpenAIClient, LLMCacheBase)

    @pytest.mark.core
    def test_invalidate_last_cache_entry(self, temp_cache_file):
        """Test that invalidate_last_cache_entry removes the last cached key and persists to JSON."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)

        # Simulate a cache entry being tracked per-thread
        client.api_cache["cache_key_1"] = {"id": "resp1", "choices": [], "created": 0, "model": "m", "object": "chat.completion"}
        client.api_cache["cache_key_2"] = {"id": "resp2", "choices": [], "created": 0, "model": "m", "object": "chat.completion"}
        client._save_cache()
        client._thread_local.last_cache_key = "cache_key_1"

        # Invalidate the last entry
        client.invalidate_last_cache_entry()

        # The entry should be gone from the in-memory cache
        assert "cache_key_1" not in client.api_cache
        assert "cache_key_2" in client.api_cache

        # Verify it was persisted to JSON on disk
        with open(temp_cache_file, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert "cache_key_1" not in on_disk
        assert "cache_key_2" in on_disk

    @pytest.mark.core
    def test_invalidate_last_cache_entry_noop_when_no_key(self, temp_cache_file):
        """Test that invalidate_last_cache_entry is a no-op when no key is tracked."""
        client = OpenAIClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        client.api_cache["key"] = {"data": "value"}
        client._save_cache()

        # No last_cache_key set — should not raise, cache should be unchanged
        client.invalidate_last_cache_entry()
        assert "key" in client.api_cache


class TestOllamaClientCache:
    """Tests for OllamaClient cache functionality via LLMCacheBase inheritance."""

    @pytest.fixture
    def temp_cache_file(self):
        """Creates a temporary file path for cache testing (file is not created)."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.mark.core
    def test_ollama_inherits_from_cache_base(self):
        """Test that OllamaClient inherits from LLMCacheBase."""
        assert issubclass(OllamaClient, LLMCacheBase)

    @pytest.mark.core
    def test_ollama_save_and_load_cache(self, temp_cache_file):
        """Test that OllamaClient can save and load cache as JSON."""
        client = OllamaClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        client.api_cache["ollama_key"] = {"content": "ollama response"}
        client._save_cache()

        assert os.path.exists(temp_cache_file)
        with open(temp_cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["ollama_key"] == {"content": "ollama response"}

    @pytest.mark.core
    def test_ollama_cache_roundtrip(self, temp_cache_file):
        """Test OllamaClient cache roundtrip via JSON."""
        client1 = OllamaClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        client1.api_cache["k1"] = {"role": "assistant", "content": "hello"}
        client1._save_cache()

        client2 = OllamaClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        assert client2.api_cache["k1"] == {"role": "assistant", "content": "hello"}

    @pytest.mark.core
    def test_ollama_set_api_cache(self, temp_cache_file):
        """Test that OllamaClient can enable/disable caching via base class."""
        client = OllamaClient(cache_api_calls=False, cache_file_name=temp_cache_file)
        assert client.cache_api_calls is False

        client.set_api_cache(True, temp_cache_file)
        assert client.cache_api_calls is True
        assert hasattr(client, 'api_cache')

    @pytest.mark.core
    def test_ollama_handles_corrupted_cache(self, temp_cache_file):
        """Test that OllamaClient gracefully handles corrupted cache files."""
        with open(temp_cache_file, "w") as f:
            f.write("not valid json")
        client = OllamaClient(cache_api_calls=True, cache_file_name=temp_cache_file)
        assert client.api_cache == {}
