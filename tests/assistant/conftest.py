"""
让 assistant tests 用独立的 SQLite 文件，避免踩到生产数据。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _temp_knowledge_db(tmp_path, monkeypatch):
    db = tmp_path / "knowledge.sqlite3"
    monkeypatch.setenv("ASSISTANT_KNOWLEDGE_DB", str(db))
    # Redirect session memory to tmp path (v6: default persistence enabled)
    monkeypatch.setenv("SESSION_MEMORY_FILE", str(tmp_path / "session_memory.json"))
    # reset default store singleton so it picks up new path
    from assistant.rag import store as _store_mod
    _store_mod._default = None
    from assistant.session_memory import reset_session_store
    reset_session_store()
    yield
    _store_mod._default = None
    reset_session_store()
