"""
投资场景情绪标签 —— 被 community_indexer 和 sentiment_aggregator 共用。

和现有 community/schema.SentimentProfile 的 5 维打分是互补关系：
  - SentimentProfile: optimism / fear / uncertainty / skepticism / hype
    针对"日报做叙事理解"
  - 这里的 EMOTION_LABELS: bullish_optimism / bearish_panic / fomo_chasing /
    uncertainty / strong_conviction / narrative_repetition / capitulation / neutral
    针对"回答用户决策问题"，标签本身就直接驱动 decision_engine 的风险修正。

两套并存不浪费 —— 前者 daily brief 还在用；后者是对话式 bot 新增。
"""
from __future__ import annotations

import re
from typing import Literal

EmotionLabel = Literal[
    "bullish_optimism",
    "bearish_panic",
    "fomo_chasing",
    "uncertainty",
    "strong_conviction",
    "narrative_repetition",
    "capitulation",
    "neutral",
]


# ─── 粗多空 ─────────────────────────────────────────────────────────────────

_BULLISH_TERMS = [
    "bullish", "long", "buy the dip", "moon", "rally", "pump", "breakout",
    "looks good", "keep buying", "to the moon", "up only", "undervalued",
    "accumulate", "add more", "strong buy", "price target raised",
    "beat expectations", "better than expected", "record high", "new high",
    "support held", "bouncing", "recovering", "oversold", "cheap here",
    "good entry", "good time to buy", "buying opportunity",
    "看多", "做多", "抄底", "加仓", "上车", "梭哈", "牛市",
    "低估", "买点", "支撑稳", "超跌", "好机会", "建仓", "布局",
]
_BEARISH_TERMS = [
    "bearish", "short", "sell", "crash", "dump", "puts", "correction",
    "rug", "bear market", "collapse", "overvalued", "overbought",
    "resistance", "extended", "don't chase", "avoid", "stay away",
    "take profit", "take gains", "reduce exposure", "cut position",
    "wait for pullback", "wait for dip", "needs to cool", "too high",
    "downside risk", "price target cut", "miss expectations", "warning",
    "missed earnings", "guidance cut", "downgrade",
    "看空", "做空", "清仓", "割肉", "跑路", "崩盘", "熊市",
    "高估", "追高风险", "压力位", "别追", "等回调", "止损", "减仓",
    "过热", "泡沫", "风险大", "不建议追",
]


def classify_bull_bear(text: str) -> tuple[str, float]:
    lower = (text or "").lower()
    bull = sum(1 for w in _BULLISH_TERMS if w in lower)
    bear = sum(1 for w in _BEARISH_TERMS if w in lower)
    if bull == 0 and bear == 0:
        return "neutral", 0.1
    if bull and bear and abs(bull - bear) <= 1:
        return "mixed", 0.3
    if bull > bear:
        conf = min(0.9, 0.3 + 0.15 * bull)
        return "bullish", conf
    conf = min(0.9, 0.3 + 0.15 * bear)
    return "bearish", conf


# ─── 投资场景情绪 ────────────────────────────────────────────────────────────

_FOMO_TERMS = [
    r"怕错过", r"别人都在", r"跟风", r"fomo", r"can'?t miss",
    r"missing out", r"来不及", r"all in now", r"别人都赚", r"大家都在买",
    r"我也冲", r"上车", r"满仓冲",
]
_PANIC_TERMS = [
    r"panic", r"crash", r"崩盘", r"闪崩", r"恐慌", r"割肉",
    r"everyone is selling", r"bloodbath", r"sell everything", r"get out",
]
_CAPITULATION_TERMS = [
    r"i give up", r"我不玩了", r"认栽", r"再也不", r"cashed out",
    r"放弃", r"done with this", r"tapping out",
]
_CONVICTION_TERMS = [
    r"all in", r"梭哈", r"重仓", r"满仓", r"diamond hands", r"钻石手",
    r"长期看好", r"长期持有", r"绝不卖", r"hold forever",
    r"i'?m confident", r"strong conviction",
]
_UNCERTAINTY_TERMS = [
    r"not sure", r"idk", r"maybe", r"could go either way",
    r"不确定", r"不知道", r"看不懂", r"拿不准", r"\?\?\?",
]
_NARRATIVE_REPETITION = [
    r"same old", r"everyone says", r"大家都说", r"重复一遍",
    r"narrative", r"故事还是那套", r"everyone agrees",
]
_OPTIMISM_TERMS = [
    r"bullish", r"long", r"看多", r"利好", r"上涨", r"rally",
    r"good time to buy", r"新高", r"breakout", r"moon",
]


def _match_count(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def classify_emotion_label(text: str) -> tuple[EmotionLabel, float]:
    """
    返回 (label, confidence 0-1)。规则优先级按"对决策影响大小"排。
    """
    scores: dict[EmotionLabel, int] = {}
    for label, patterns in [
        ("fomo_chasing", _FOMO_TERMS),
        ("bearish_panic", _PANIC_TERMS),
        ("capitulation", _CAPITULATION_TERMS),
        ("strong_conviction", _CONVICTION_TERMS),
        ("narrative_repetition", _NARRATIVE_REPETITION),
        ("uncertainty", _UNCERTAINTY_TERMS),
        ("bullish_optimism", _OPTIMISM_TERMS),
    ]:
        c = _match_count(text, patterns)
        if c:
            scores[label] = c  # type: ignore[index]

    if not scores:
        return "neutral", 0.1

    label, count = max(scores.items(), key=lambda kv: kv[1])
    return label, min(0.9, 0.3 + 0.15 * count)
