"""Tests for company_context module."""
from __future__ import annotations

import json
import pytest
from assistant.company_context import (
    CompanyContext,
    DEFAULT_COMPANY_CONTEXT,
    get_company_context,
    reset_company_context,
    _load_from_file,
)
from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_company_context()
    yield
    reset_company_context()


class TestCompanyContext:
    def test_default_context_has_required_fields(self):
        ctx = DEFAULT_COMPANY_CONTEXT
        assert ctx.company_name
        assert ctx.business_type
        assert ctx.risk_appetite in ("aggressive", "moderate", "conservative")
        assert len(ctx.focus_assets) > 0
        assert len(ctx.banned_phrases) > 0
        assert ctx.analyst_persona

    def test_to_system_block_contains_key_fields(self):
        ctx = DEFAULT_COMPANY_CONTEXT
        block = ctx.to_system_block()
        assert "公司语境" in block
        assert ctx.business_type in block
        assert ctx.risk_appetite in block

    def test_is_focus_asset_known(self):
        ctx = DEFAULT_COMPANY_CONTEXT
        assert ctx.is_focus_asset("gold") is True
        assert ctx.is_focus_asset("bitcoin") is True
        assert ctx.is_focus_asset("xyz_unknown") is False
        assert ctx.is_focus_asset(None) is False

    def test_has_banned_phrase_detection(self):
        ctx = DEFAULT_COMPANY_CONTEXT
        # 明确的禁用短语
        assert "取决于你的风险偏好" in ctx.has_banned_phrase(
            "这取决于你的风险偏好，请自行决定"
        )
        # 没有禁用短语
        assert ctx.has_banned_phrase("今天黄金上涨，信号偏多") == []

    def test_to_dict_is_serializable(self):
        ctx = DEFAULT_COMPANY_CONTEXT
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert d["company_name"] == ctx.company_name
        assert isinstance(d["focus_assets"], list)
        assert isinstance(d["banned_phrases"], list)

    def test_get_company_context_returns_default(self, monkeypatch):
        monkeypatch.delenv("COMPANY_CONTEXT_FILE", raising=False)
        ctx = get_company_context()
        assert ctx is not None
        assert ctx.company_name

    def test_load_from_file(self, tmp_path):
        data = {
            "company_name": "TestCo",
            "business_type": "hedge_fund",
            "risk_appetite": "aggressive",
            "focus_assets": ["gold", "bitcoin"],
            "focus_themes": ["macro"],
            "preferred_output_style": "concise_analytical",
            "banned_phrases": ["请咨询"],
            "analyst_persona": "测试分析师",
            "internal_context": "测试环境",
        }
        f = tmp_path / "ctx.json"
        f.write_text(json.dumps(data))
        ctx = _load_from_file(f)
        assert ctx is not None
        assert ctx.company_name == "TestCo"
        assert ctx.risk_appetite == "aggressive"
        assert "gold" in ctx.focus_assets

    def test_load_from_env(self, tmp_path, monkeypatch):
        data = {"company_name": "EnvCo", "business_type": "trading_desk"}
        f = tmp_path / "env_ctx.json"
        f.write_text(json.dumps(data))
        monkeypatch.setenv("COMPANY_CONTEXT_FILE", str(f))
        ctx = get_company_context()
        assert ctx.company_name == "EnvCo"


class TestCompanyContextBannedPhrases:
    def test_banned_phrase_case_insensitive(self):
        ctx = CompanyContext(
            banned_phrases=("This is not financial advice",),
        )
        found = ctx.has_banned_phrase("THIS IS NOT FINANCIAL ADVICE to anyone")
        assert len(found) > 0

    def test_multiple_banned_phrases_detected(self):
        ctx = CompanyContext(
            banned_phrases=("phrase one", "phrase two"),
        )
        result = ctx.has_banned_phrase("phrase one and also phrase two appear here")
        assert len(result) == 2
