"""
Unit tests for clustering._embed_batch with Ollama.
Both chat/LLM and embeddings go through Ollama on port 11434.
All HTTP calls are monkeypatched — no real server needed.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _ollama_embed_response(vector: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embedding": vector}
    resp.raise_for_status = MagicMock()
    return resp


class TestEmbedBatch:
    def test_calls_ollama_api_embeddings(self):
        import community.clustering as mod
        orig_base = mod.OLLAMA_API_BASE
        orig_model = mod.OLLAMA_EMBED_MODEL
        mod.OLLAMA_API_BASE = "http://localhost:11434/v1"
        mod.OLLAMA_EMBED_MODEL = "nomic-embed-text"
        try:
            vec = [0.1, 0.2, 0.3]
            with patch("requests.post", return_value=_ollama_embed_response(vec)) as mock_post:
                result = mod._embed_batch(["hello world"])
            url = mock_post.call_args[0][0]
            assert url == "http://localhost:11434/api/embeddings"
            body = mock_post.call_args[1]["json"]
            assert body["model"] == "nomic-embed-text"
            assert body["prompt"] == "hello world"
            assert result == [vec]
        finally:
            mod.OLLAMA_API_BASE = orig_base
            mod.OLLAMA_EMBED_MODEL = orig_model

    def test_strips_v1_suffix_from_base(self):
        import community.clustering as mod
        orig_base = mod.OLLAMA_API_BASE
        mod.OLLAMA_API_BASE = "http://localhost:11434/v1"
        try:
            with patch("requests.post", return_value=_ollama_embed_response([0.1])) as mock_post:
                mod._embed_batch(["x"])
            assert mock_post.call_args[0][0] == "http://localhost:11434/api/embeddings"
        finally:
            mod.OLLAMA_API_BASE = orig_base

    def test_returns_none_on_connection_error(self):
        import community.clustering as mod
        orig_warned = mod._EMBED_FALLBACK_WARNED
        mod._EMBED_FALLBACK_WARNED = False
        try:
            with patch("requests.post", side_effect=ConnectionError("refused")):
                result = mod._embed_batch(["text"])
            assert result is None
        finally:
            mod._EMBED_FALLBACK_WARNED = orig_warned

    def test_empty_input_returns_empty_list(self):
        import community.clustering as mod
        assert mod._embed_batch([]) == []

    def test_multiple_texts_batched_individually(self):
        import community.clustering as mod
        v1, v2 = [1.0, 0.0], [0.0, 1.0]
        responses = [_ollama_embed_response(v1), _ollama_embed_response(v2)]
        with patch("requests.post", side_effect=responses):
            result = mod._embed_batch(["text a", "text b"])
        assert result == [v1, v2]
