"""
Company Context —— 公司语境注入层。

让 bot 知道"我代表谁说话"。每次回答前自动注入，影响：
  - 最终回复的措辞风格（偏专业 / 偏散户友好）
  - 禁止话术（某些公司不想看到的表达）
  - 关注资产重点（有助于模糊查询的优先级）
  - 风险偏好（影响 reply_composer 的建仓激进程度描述）

不影响 decision_engine 的信号计算——那部分保持纯市场逻辑。

加载顺序：
  1. COMPANY_CONTEXT_FILE 环境变量指向的 JSON 文件
  2. 项目根 company_context.json
  3. DEFAULT_COMPANY_CONTEXT（内置 demo 语境）
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── 数据类 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompanyContext:
    company_name: str = "InHouse Capital"
    business_type: str = "trading_desk"        # hedge_fund | family_office | trading_desk | research_firm | retail_platform
    risk_appetite: str = "moderate"            # aggressive | moderate | conservative
    focus_assets: tuple[str, ...] = ("gold", "bitcoin", "sp500", "nvidia", "oil")
    focus_themes: tuple[str, ...] = ("macro", "commodities", "crypto", "ai_equities")
    preferred_output_style: str = "concise_analytical"  # concise_analytical | narrative | structured_bullets
    banned_phrases: tuple[str, ...] = (
        "取决于你的风险偏好",
        "建议咨询专业人士",
        "请自行判断",
        "仅供参考",
        "不构成投资建议",
        "this is not financial advice",
        "past performance is not indicative",
        "depends on your risk appetite",
        "depends on your risk tolerance",
        "consult a professional",
        "consult a financial advisor",
        "please judge for yourself",
        "for reference only",
        "not financial advice",
        "past performance is not",
    )
    analyst_persona: str = (
        "你是一名有实战经验的市场分析师，"
        "风格直接、有立场、有根据。"
        "不说废话，先给结论，再讲依据。"
    )
    internal_context: str = ""  # 额外的背景说明（公司特定情况）

    def to_system_block(self) -> str:
        """生成注入 LLM prompt 的公司语境段。"""
        lines = [
            f"[公司语境]",
            f"机构类型：{self.business_type}，风险偏好：{self.risk_appetite}",
            f"重点关注资产：{', '.join(self.focus_assets[:5])}",
            f"主题重心：{', '.join(self.focus_themes[:4])}",
            f"风格要求：{self.preferred_output_style}",
            f"分析师定位：{self.analyst_persona}",
        ]
        if self.internal_context:
            lines.append(f"背景：{self.internal_context}")
        return "\n".join(lines)

    def is_focus_asset(self, asset: str | None) -> bool:
        return asset in self.focus_assets if asset else False

    def has_banned_phrase(self, text: str) -> list[str]:
        """返回出现在 text 中的禁用短语列表（用于校验）。"""
        lower = text.lower()
        return [p for p in self.banned_phrases if p.lower() in lower]

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_name": self.company_name,
            "business_type": self.business_type,
            "risk_appetite": self.risk_appetite,
            "focus_assets": list(self.focus_assets),
            "focus_themes": list(self.focus_themes),
            "preferred_output_style": self.preferred_output_style,
            "banned_phrases": list(self.banned_phrases),
            "analyst_persona": self.analyst_persona,
            "internal_context": self.internal_context,
        }


# ─── 默认语境（demo 场景：一个宏观交易台） ───────────────────────────────────

DEFAULT_COMPANY_CONTEXT = CompanyContext(
    company_name="InHouse Capital",
    business_type="trading_desk",
    risk_appetite="moderate",
    focus_assets=("gold", "bitcoin", "sp500", "nvidia", "oil"),
    focus_themes=("macro", "commodities", "crypto", "ai_equities"),
    preferred_output_style="concise_analytical",
    banned_phrases=(
        "取决于你的风险偏好",
        "建议咨询专业人士",
        "请自行判断",
        "仅供参考",
        "不构成投资建议",
        "this is not financial advice",
        "past performance is not",
        "depends on your risk appetite",
        "depends on your risk tolerance",
        "consult a professional",
        "consult a financial advisor",
        "please judge for yourself",
        "for reference only",
        "not financial advice",
    ),
    analyst_persona=(
        "你是一名有实战经验的宏观市场分析师，"
        "直接给出你的判断，先结论后依据，"
        "引用真实证据，不说免责话术。"
    ),
    internal_context="专注全球宏观 + 大宗商品 + 科技成长股",
)


# ─── 加载器 ──────────────────────────────────────────────────────────────────

def _load_from_file(path: Path) -> CompanyContext | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CompanyContext(
            company_name=data.get("company_name", DEFAULT_COMPANY_CONTEXT.company_name),
            business_type=data.get("business_type", DEFAULT_COMPANY_CONTEXT.business_type),
            risk_appetite=data.get("risk_appetite", DEFAULT_COMPANY_CONTEXT.risk_appetite),
            focus_assets=tuple(data.get("focus_assets", DEFAULT_COMPANY_CONTEXT.focus_assets)),
            focus_themes=tuple(data.get("focus_themes", DEFAULT_COMPANY_CONTEXT.focus_themes)),
            preferred_output_style=data.get("preferred_output_style", DEFAULT_COMPANY_CONTEXT.preferred_output_style),
            banned_phrases=tuple(data.get("banned_phrases", DEFAULT_COMPANY_CONTEXT.banned_phrases)),
            analyst_persona=data.get("analyst_persona", DEFAULT_COMPANY_CONTEXT.analyst_persona),
            internal_context=data.get("internal_context", DEFAULT_COMPANY_CONTEXT.internal_context),
        )
    except Exception as e:
        logger.warning("Failed to load company context from %s: %s", path, e)
        return None


_cached: CompanyContext | None = None


def get_company_context() -> CompanyContext:
    """返回公司语境（带缓存）。每次进程内只加载一次。"""
    global _cached
    if _cached is not None:
        return _cached

    # 1. 环境变量指定的文件
    env_path = os.getenv("COMPANY_CONTEXT_FILE", "").strip()
    if env_path:
        ctx = _load_from_file(Path(env_path))
        if ctx:
            _cached = ctx
            return _cached

    # 2. 项目根 company_context.json
    default_file = Path(__file__).resolve().parents[2] / "company_context.json"
    if default_file.exists():
        ctx = _load_from_file(default_file)
        if ctx:
            _cached = ctx
            return _cached

    # 3. 内置默认
    _cached = DEFAULT_COMPANY_CONTEXT
    return _cached


def reset_company_context() -> None:
    """测试钩子：强制重新加载。"""
    global _cached
    _cached = None
