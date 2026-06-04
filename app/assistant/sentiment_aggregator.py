"""
社区情绪聚合器 —— 把若干条带 emotion_label 的 community 帖子，
按资产聚合成一个"投资场景上的情绪读数"。

这是 decision_engine 的主要输入之一。和 community/schema 的
CommunityAnalystReport 区别在于：
  - CommunityAnalystReport 面向日报（叙事 + 保险角度）
  - AggregatedSentiment 面向用户决策（"该不该买"）—— 给多空比、
    FOMO 比例、叙事关键词、拥挤交易风险
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Iterable

from assistant.asset_taxonomy import asset_display
from assistant.rag.store import CommunityDoc

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSentiment:
    asset: str | None
    window: str                         # e.g. "3d"
    post_count: int
    overall_bias: str                   # bullish | bearish | neutral | mixed
    bullish_ratio: float
    bearish_ratio: float
    fomo_ratio: float
    uncertainty_ratio: float
    conviction_ratio: float
    panic_ratio: float
    narrative_keywords: list[str] = field(default_factory=list)
    crowded_trade_risk: str = "low"     # low | medium | high
    summary: str = ""
    platforms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── 叙事关键词提取（极简版） ───────────────────────────────────────────────

_NARRATIVE_CANDIDATES = [
    "safe haven", "rate cut", "rate hike", "inflation", "hedge",
    "fomo", "all in", "follow the rally", "breakout", "new high",
    "avalanche of buys", "short squeeze", "buy the dip",
    "降息", "加息", "通胀", "避险", "上车", "抄底", "跟风", "新高",
    "战争", "地缘", "衰退",
]


def _extract_narratives(docs: list[CommunityDoc], top_n: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for d in docs:
        lower = d.raw_text.lower()
        for kw in _NARRATIVE_CANDIDATES:
            if kw.lower() in lower:
                counter[kw] += 1
    return [kw for kw, _ in counter.most_common(top_n)]


# ─── 拥挤交易风险 ────────────────────────────────────────────────────────────

def _crowded_trade_risk(bullish_ratio: float, fomo_ratio: float,
                       conviction_ratio: float) -> str:
    # "大家都看多 + 一堆人 FOMO + 一堆人 all in" —— 拥挤
    if bullish_ratio >= 0.80 and (fomo_ratio >= 0.30 or conviction_ratio >= 0.30):
        return "high"
    if bullish_ratio >= 0.65 and fomo_ratio >= 0.20:
        return "medium"
    return "low"


# ─── 聚合主函数 ──────────────────────────────────────────────────────────────

def aggregate(
    docs: Iterable[CommunityDoc],
    asset: str | None = None,
    window_hours: int = 72,
) -> AggregatedSentiment:
    docs_list = list(docs)
    n = len(docs_list)

    if n == 0:
        return AggregatedSentiment(
            asset=asset,
            window=f"{window_hours // 24}d" if window_hours >= 24 else f"{window_hours}h",
            post_count=0,
            overall_bias="neutral",
            bullish_ratio=0.0, bearish_ratio=0.0,
            fomo_ratio=0.0, uncertainty_ratio=0.0,
            conviction_ratio=0.0, panic_ratio=0.0,
            narrative_keywords=[],
            crowded_trade_risk="low",
            summary=f"最近没有拿到 {asset_display(asset)} 相关的社区讨论。",
            platforms=[],
        )

    # 多空分布。bullish_bearish_label 在社区帖子里经常是中性（没直说"看多/看空"），
    # 但投资场景情绪标签本身就能告诉我们方向：
    #   fomo_chasing / strong_conviction -> 实质上是多头
    #   bearish_panic / capitulation     -> 实质上是空头（或抛压）
    # 如果显式多空标签是 mixed/neutral 但情绪标签明确，就用情绪标签推断方向。
    def _effective_bb(d: CommunityDoc) -> str:
        if d.bullish_bearish_label in ("bullish", "bearish"):
            return d.bullish_bearish_label
        if d.emotion_label in ("fomo_chasing", "strong_conviction", "bullish_optimism"):
            return "bullish"
        if d.emotion_label in ("bearish_panic", "capitulation"):
            return "bearish"
        return d.bullish_bearish_label  # neutral / mixed

    bb_counter = Counter(_effective_bb(d) for d in docs_list)
    bullish = bb_counter.get("bullish", 0)
    bearish = bb_counter.get("bearish", 0)
    bullish_ratio = bullish / n
    bearish_ratio = bearish / n

    if bullish_ratio >= 0.55 and bullish_ratio > bearish_ratio + 0.1:
        overall_bias = "bullish"
    elif bearish_ratio >= 0.55 and bearish_ratio > bullish_ratio + 0.1:
        overall_bias = "bearish"
    elif abs(bullish_ratio - bearish_ratio) < 0.15 and (bullish or bearish):
        overall_bias = "mixed"
    else:
        overall_bias = "neutral"

    # 投资场景情绪分布
    em_counter = Counter(d.emotion_label for d in docs_list)
    fomo_ratio = em_counter.get("fomo_chasing", 0) / n
    uncertainty_ratio = em_counter.get("uncertainty", 0) / n
    conviction_ratio = em_counter.get("strong_conviction", 0) / n
    panic_ratio = em_counter.get("bearish_panic", 0) / n

    narratives = _extract_narratives(docs_list)
    crowded = _crowded_trade_risk(bullish_ratio, fomo_ratio, conviction_ratio)

    platforms = sorted(set(d.platform for d in docs_list if d.platform))

    summary = _summarize(
        asset=asset,
        overall_bias=overall_bias,
        bullish_ratio=bullish_ratio,
        bearish_ratio=bearish_ratio,
        fomo_ratio=fomo_ratio,
        conviction_ratio=conviction_ratio,
        panic_ratio=panic_ratio,
        crowded=crowded,
        n=n,
    )

    return AggregatedSentiment(
        asset=asset,
        window=f"{window_hours // 24}d" if window_hours >= 24 else f"{window_hours}h",
        post_count=n,
        overall_bias=overall_bias,
        bullish_ratio=round(bullish_ratio, 2),
        bearish_ratio=round(bearish_ratio, 2),
        fomo_ratio=round(fomo_ratio, 2),
        uncertainty_ratio=round(uncertainty_ratio, 2),
        conviction_ratio=round(conviction_ratio, 2),
        panic_ratio=round(panic_ratio, 2),
        narrative_keywords=narratives,
        crowded_trade_risk=crowded,
        summary=summary,
        platforms=platforms,
    )


def _summarize(
    asset: str | None,
    overall_bias: str,
    bullish_ratio: float,
    bearish_ratio: float,
    fomo_ratio: float,
    conviction_ratio: float,
    panic_ratio: float,
    crowded: str,
    n: int,
) -> str:
    name = asset_display(asset)
    if overall_bias == "bullish":
        base = f"社区对 {name} 整体偏看多（bullish {bullish_ratio:.0%}）"
    elif overall_bias == "bearish":
        base = f"社区对 {name} 整体偏看空（bearish {bearish_ratio:.0%}）"
    elif overall_bias == "mixed":
        base = f"社区对 {name} 分歧较大（多 {bullish_ratio:.0%} / 空 {bearish_ratio:.0%}）"
    else:
        base = f"社区对 {name} 讨论较少或情绪偏中性（{n} 条）"

    extras: list[str] = []
    if fomo_ratio >= 0.25:
        extras.append(f"有明显跟风情绪（FOMO {fomo_ratio:.0%}）")
    if conviction_ratio >= 0.25:
        extras.append(f"重仓/梭哈声量不小（{conviction_ratio:.0%}）")
    if panic_ratio >= 0.2:
        extras.append(f"出现恐慌性抛售言论（{panic_ratio:.0%}）")
    if crowded == "high":
        extras.append("存在明显拥挤交易迹象")
    elif crowded == "medium":
        extras.append("有一定拥挤迹象，需警惕一致预期反转")

    if extras:
        return base + "，" + "；".join(extras) + "。"
    return base + "。"


# ─── 便捷入口：直接用 retriever 拉数据再聚合 ─────────────────────────────────

def aggregate_for_asset(
    asset: str | None,
    window_hours: int = 72,
    top_k: int = 200,
) -> AggregatedSentiment:
    """一步到位：检索 -> 聚合。供 pipeline 调用。"""
    from assistant.rag.retriever import Retriever
    docs = Retriever().retrieve_community(
        asset=asset, window_hours=window_hours, top_k=top_k,
    )
    return aggregate(docs, asset=asset, window_hours=window_hours)
