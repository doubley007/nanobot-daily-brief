"""
用户情绪理解器。

Query Router 给出的 user_emotion 是粗枚举，用于路由决策；
这个模块在"已确认进入某条链路后"给出更细的心理画像，供 reply_composer
调整措辞（比如是否先共情再给建议、是否强调"别重仓追"等）。

输出结构化 UserEmotionProfile，而不是 free text。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Callable, Literal

logger = logging.getLogger(__name__)


PrimaryEmotion = Literal[
    "anxious", "fomo", "uncertain", "frustrated",
    "confident", "seeking_confirmation", "neutral",
]


@dataclass
class UserEmotionProfile:
    primary_emotion: PrimaryEmotion
    emotion_intensity: float          # 0.0 - 1.0
    needs_confirmation: bool
    risk_of_impulsive_action: bool
    signals: list[str]                # 被触发的规则关键词，便于调试
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─── 规则组：每组含触发词、对应情绪、强度加分 ────────────────────────────────

_RULES: list[tuple[PrimaryEmotion, list[str], float]] = [
    (
        "fomo",
        [r"怕错过", r"别人都在", r"大家都在", r"跟风", r"是不是也该",
         r"来不及", r"fomo", r"missing out", r"上车", r"别人都赚"],
        0.3,
    ),
    (
        "frustrated",
        [r"亏麻", r"亏惨", r"亏了很多", r"套住", r"割肉", r"心态崩",
         r"崩溃", r"i lost", r"down \d+%", r"regret"],
        0.35,
    ),
    (
        "anxious",
        [r"焦虑", r"好慌", r"睡不着", r"担心", r"怕", r"scared",
         r"stressed", r"anxious", r"worry"],
        0.25,
    ),
    (
        "seeking_confirmation",
        [r"大家都看多", r"大家都看好", r"都说能涨", r"是不是该",
         r"对不对", r"我这样想对吗", r"应该没问题吧", r"right\?"],
        0.25,
    ),
    (
        "uncertain",
        [r"不确定", r"犹豫", r"拿不准", r"要不要", r"该不该",
         r"not sure", r"should i", r"should I"],
        0.2,
    ),
    (
        "confident",
        [r"我觉得一定", r"肯定涨", r"稳了", r"铁定", r"一定会",
         r"definitely", r"for sure", r"all in"],
        0.3,
    ),
]


_IMPULSIVE_SIGNALS = [
    r"all in", r"梭哈", r"一把梭", r"重仓", r"满仓", r"借钱买",
    r"杠杆", r"leverage", r"不管多少", r"有多少买多少",
]


def _collect_matches(text: str) -> dict[PrimaryEmotion, tuple[float, list[str]]]:
    """按规则组统计命中强度和触发词。"""
    results: dict[PrimaryEmotion, tuple[float, list[str]]] = {}
    for emotion, patterns, weight in _RULES:
        hits: list[str] = []
        score = 0.0
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                hits.append(m.group(0))
                score += weight
        if hits:
            results[emotion] = (score, hits)
    return results


def _rule_based_profile(text: str) -> UserEmotionProfile:
    matches = _collect_matches(text)

    if not matches:
        return UserEmotionProfile(
            primary_emotion="neutral",
            emotion_intensity=0.1,
            needs_confirmation=False,
            risk_of_impulsive_action=False,
            signals=[],
            notes="no trigger matched",
        )

    primary, (score, hits) = max(matches.items(), key=lambda kv: kv[1][0])
    intensity = min(1.0, 0.2 + score)

    needs_confirmation = (
        "seeking_confirmation" in matches
        or primary in ("fomo", "uncertain")
    )
    has_impulsive_words = any(
        re.search(p, text, flags=re.IGNORECASE) for p in _IMPULSIVE_SIGNALS
    )
    risk_of_impulsive = has_impulsive_words or (primary == "fomo" and intensity >= 0.5)

    all_signals: list[str] = []
    for _, h in matches.values():
        all_signals.extend(h)

    return UserEmotionProfile(
        primary_emotion=primary,
        emotion_intensity=round(intensity, 2),
        needs_confirmation=needs_confirmation,
        risk_of_impulsive_action=risk_of_impulsive,
        signals=all_signals,
        notes="rule-based",
    )


# ─── LLM 兜底：规则没命中时让 LLM 给一轮判断 ─────────────────────────────────

_LLM_EMOTION_PROMPT = """You are analysing the psychological state of an investor from a message they just sent.
This drives reply tone later (neither over-soothing nor cold).

Output JSON only, strict schema:
{
  "primary_emotion": "anxious|fomo|uncertain|frustrated|confident|seeking_confirmation|neutral",
  "emotion_intensity": 0.0-1.0,
  "needs_confirmation": true|false,
  "risk_of_impulsive_action": true|false,
  "notes": "one short sentence explaining the judgment"
}

Respond in English.

User message:
\"\"\"{text}\"\"\"
"""


def _llm_profile(text: str, llm_callable: Callable[[str], str]) -> UserEmotionProfile | None:
    try:
        raw = llm_callable(_LLM_EMOTION_PROMPT.format(text=text))
        data = json.loads(raw)
    except Exception as e:
        logger.warning("User-emotion LLM fallback failed: %s", e)
        return None

    emotion = data.get("primary_emotion", "neutral")
    allowed = {"anxious", "fomo", "uncertain", "frustrated",
               "confident", "seeking_confirmation", "neutral"}
    if emotion not in allowed:
        emotion = "neutral"

    try:
        intensity = float(data.get("emotion_intensity", 0.3))
    except (TypeError, ValueError):
        intensity = 0.3
    intensity = max(0.0, min(1.0, intensity))

    return UserEmotionProfile(
        primary_emotion=emotion,  # type: ignore[arg-type]
        emotion_intensity=round(intensity, 2),
        needs_confirmation=bool(data.get("needs_confirmation", False)),
        risk_of_impulsive_action=bool(data.get("risk_of_impulsive_action", False)),
        signals=[],
        notes=str(data.get("notes", ""))[:200] or "llm",
    )


# ─── 对外入口 ────────────────────────────────────────────────────────────────

def analyze_user_emotion(
    text: str,
    llm_callable: Callable[[str], str] | None = None,
) -> UserEmotionProfile:
    """
    规则 + LLM 混合：规则命中则以规则为准（更稳定、可解释），
    规则没命中再用 LLM，LLM 也不可用就返回 neutral。
    """
    text = (text or "").strip()
    if not text:
        return UserEmotionProfile(
            primary_emotion="neutral", emotion_intensity=0.0,
            needs_confirmation=False, risk_of_impulsive_action=False,
            signals=[], notes="empty",
        )

    rule = _rule_based_profile(text)
    if rule.primary_emotion != "neutral" and rule.emotion_intensity >= 0.3:
        return rule

    if llm_callable is not None:
        llm = _llm_profile(text, llm_callable)
        if llm is not None:
            return llm

    return rule
