"""
Query Router —— 消息路由。

把 Telegram 用户发来的一段自然语言分类成三条链路之一：

  emotional_chat    用户主要在表达情绪，没有明确的标的或决策问题
  market_decision   用户在问"能不能买/要不要卖/还能追吗" —— 需要跑决策引擎
  market_summary    用户在问"今天/最近市场怎么样" —— 需要跑汇总

设计要点：
  1. 规则优先。规则能一眼看出来的，直接短路，不浪费 LLM 调用。
  2. LLM 兜底。规则拿不准（例如中文表达非常口语化、没有疑问词）时，让 LLM
     按结构化输出给答案，再映射回枚举值。
  3. 输出一个确定的 dataclass，下游不用再做兼容判断。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Callable, Literal

from assistant.asset_taxonomy import detect_asset

logger = logging.getLogger(__name__)


Route = Literal["emotional_chat", "market_decision", "market_summary"]
UserEmotion = Literal["anxious", "fomo", "uncertain", "frustrated", "neutral"]


@dataclass
class RouterResult:
    route: Route
    asset: str | None
    # ── TONE ONLY ──────────────────────────────────────────────────────────────
    # user_emotion describes how the *user* is feeling, derived from their
    # message text.  It flows to reply_composer to adjust tone/openers/warnings.
    # It must NEVER be used as a market signal inside decision_engine or
    # sentiment_aggregator — those modules read CommunityDoc.emotion_label,
    # which is derived from third-party social posts, not from this user.
    # ──────────────────────────────────────────────────────────────────────────
    user_emotion: UserEmotion
    confidence: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─── 规则词典 ────────────────────────────────────────────────────────────────

_DECISION_PATTERNS = [
    r"能不能买", r"可以买", r"要不要买", r"该不该买", r"值不值得", r"能追吗",
    r"还能上车", r"还能追", r"该加仓", r"要不要卖", r"要不要清", r"该止损",
    r"抄底", r"能不能入", r"can i buy", r"should i buy", r"worth buying",
    r"good time to buy", r"加仓", r"减仓",
    # 更口语化的决策表达
    r"怎么看", r"后市", r"还能涨", r"还会涨", r"还有空间", r"还有多少空间",
    r"分析一下", r"看一下", r"帮我看看", r"适合买", r"可以入场", r"值得入场",
    r"能补仓", r"要止盈", r"要清仓", r"能涨多少", r"会不会跌", r"会跌吗",
    r"还能持有", r"值得持有", r"继续拿", r"还拿得住", r"现在怎么样",
    r"买点", r"卖点", r"支撑", r"压力位", r"目标价",
    # 宏观/利率/保险类问题 — 无资产名称也属于决策类
    r"国债收益率", r"收益率", r"利差", r"credit spread", r"信用利差",
    r"宏观环境", r"保险公司", r"投资组合", r"组合影响", r"配置",
    r"应该减少", r"该减少", r"应该增加", r"该增加", r"该配置",
    r"有什么影响", r"影响.*组合", r"对.*影响",
    r"bond yield", r"treasury yield", r"interest rate",
]

_SUMMARY_PATTERNS = [
    r"今天.*怎么了", r"最近.*怎么样", r"大家.*讨论", r"市场.*总结",
    r"情绪怎么样", r"什么新闻", r"what'?s happening", r"market summary",
    r"最近.*行情", r"新闻总结",
]

_EMOTIONAL_PATTERNS = [
    r"焦虑", r"难受", r"亏麻", r"亏惨", r"睡不着", r"怕错过", r"好慌",
    r"我完了", r"崩溃", r"心态崩", r"stressed", r"anxious", r"i'?m scared",
    r"i lost", r"regret",
]

_FOMO_PATTERNS = [
    r"怕错过", r"别人都在", r"跟风", r"是不是该上", r"我是不是也", r"来不及",
    r"fomo", r"missing out", r"jump in", r"上车",
]

_FRUSTRATED_PATTERNS = [
    r"亏麻", r"亏惨", r"亏了很多", r"套住", r"割肉", r"崩溃", r"心态崩",
    r"i lost", r"down (\d+)%",
]

_ANXIOUS_PATTERNS = [
    r"焦虑", r"好慌", r"睡不着", r"怕", r"担心", r"stressed", r"anxious",
    r"scared", r"worry", r"worried",
]

_UNCERTAIN_PATTERNS = [
    r"是不是", r"该不该", r"要不要", r"不确定", r"犹豫", r"拿不准",
    r"not sure", r"should i",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


# ─── 规则层 ──────────────────────────────────────────────────────────────────

def _rule_based_emotion(text: str) -> UserEmotion:
    if _match_any(text, _FOMO_PATTERNS):
        return "fomo"
    if _match_any(text, _FRUSTRATED_PATTERNS):
        return "frustrated"
    if _match_any(text, _ANXIOUS_PATTERNS):
        return "anxious"
    if _match_any(text, _UNCERTAIN_PATTERNS):
        return "uncertain"
    return "neutral"


def _rule_based_route(text: str, asset: str | None) -> tuple[Route | None, float]:
    """
    返回 (route, confidence)。如果规则判不准就返回 (None, 0)。
    """
    has_decision_cue = _match_any(text, _DECISION_PATTERNS)
    has_summary_cue = _match_any(text, _SUMMARY_PATTERNS)
    has_emotional_cue = _match_any(text, _EMOTIONAL_PATTERNS)

    # 明确询问某资产 + 动作词 -> market_decision
    if asset and has_decision_cue:
        return "market_decision", 0.92

    # 资产 + 问号疑问但没动作词，也多半是决策类
    if asset and ("?" in text or "？" in text or _match_any(text, _UNCERTAIN_PATTERNS)):
        return "market_decision", 0.78

    # 宏观/专业问题有 decision cue 但无资产（e.g. 收益率、利差、保险配置）
    if has_decision_cue and not has_emotional_cue:
        return "market_decision", 0.80

    # 明确的"大盘/最近/情绪"类 -> summary
    if has_summary_cue and not has_decision_cue:
        return "market_summary", 0.85

    # 纯情绪宣泄，且没有资产/没有动作 -> emotional_chat
    if has_emotional_cue and not has_decision_cue and not asset:
        return "emotional_chat", 0.82

    return None, 0.0


# ─── LLM 兜底 ────────────────────────────────────────────────────────────────

_LLM_ROUTER_PROMPT = """You are a message classifier. Classify the user message below.

Enums (use exactly these):
  route:
    - emotional_chat   User is mainly venting / expressing feelings; no explicit ask for a decision on a specific asset.
    - market_decision  User is asking whether to participate in an asset (buy / sell / add / stop-loss).
    - market_summary   User is asking about the market in general, a summary, or recent themes.
  asset: If you can identify one of the allowed assets below, return its id; else null.
    Allowed: gold, bitcoin, ethereum, tesla, nvidia, sp500, nasdaq, usd, oil, silver, copper,
             a_shares, hk_stocks, sti, dbs, ocbc, uob, cict, mapletree_pan_asia, sgd
  user_emotion: anxious | fomo | uncertain | frustrated | neutral

Output JSON only. Respond in English (rationale in English):
{"route": "...", "asset": "...", "user_emotion": "...", "confidence": 0.xx, "rationale": "..."}

User message:
\"\"\"{text}\"\"\"
"""


def _llm_route(text: str, llm_callable: Callable[[str], str]) -> RouterResult | None:
    try:
        raw = llm_callable(_LLM_ROUTER_PROMPT.format(text=text))
        data = json.loads(raw)
    except Exception as e:
        logger.warning("Router LLM fallback failed: %s", e)
        return None

    route = data.get("route", "emotional_chat")
    if route not in ("emotional_chat", "market_decision", "market_summary"):
        route = "emotional_chat"
    emotion = data.get("user_emotion", "neutral")
    if emotion not in ("anxious", "fomo", "uncertain", "frustrated", "neutral"):
        emotion = "neutral"
    asset = data.get("asset")
    if asset == "null":
        asset = None
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return RouterResult(
        route=route,  # type: ignore[arg-type]
        asset=asset,
        user_emotion=emotion,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, confidence)),
        rationale=str(data.get("rationale", ""))[:200],
    )


# ─── 对外入口 ────────────────────────────────────────────────────────────────

def route_query(
    text: str,
    llm_callable: Callable[[str], str] | None = None,
) -> RouterResult:
    """
    混合策略：先跑规则，命中就直接用；否则调 LLM 兜底；LLM 也失败就 neutral。
    """
    text = (text or "").strip()
    if not text:
        return RouterResult(
            route="emotional_chat", asset=None,
            user_emotion="neutral", confidence=0.1,
            rationale="empty input",
        )

    asset = detect_asset(text)
    rule_route, rule_conf = _rule_based_route(text, asset)
    rule_emotion = _rule_based_emotion(text)

    # 规则够自信直接返回
    if rule_route and rule_conf >= 0.75:
        return RouterResult(
            route=rule_route,
            asset=asset,
            user_emotion=rule_emotion,
            confidence=rule_conf,
            rationale="rule-based",
        )

    # 规则拿不准，且有 LLM 可用 —— 让 LLM 仲裁
    if llm_callable is not None:
        llm_res = _llm_route(text, llm_callable)
        if llm_res is not None:
            # 规则识别到了资产但 LLM 没识别，保留规则的
            if asset and not llm_res.asset:
                llm_res.asset = asset
            # 规则识别到的情绪不是 neutral 优先保留
            if rule_emotion != "neutral" and llm_res.user_emotion == "neutral":
                llm_res.user_emotion = rule_emotion
            return llm_res

    # LLM 也没有：保守兜底
    fallback_route: Route = (
        "market_decision" if asset else
        ("market_summary" if rule_route == "market_summary" else "emotional_chat")
    )
    return RouterResult(
        route=fallback_route,
        asset=asset,
        user_emotion=rule_emotion,
        confidence=max(rule_conf, 0.4),
        rationale="rule-fallback (no LLM)",
    )
