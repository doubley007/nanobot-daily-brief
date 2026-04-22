"""
End-to-end sample brief for the community-analysis v2 formatter.

Builds a realistic set of TopicClusters with the new TrendProfile +
InsuranceFramework fields populated, runs format_community_section, and
prints both BEFORE and AFTER blocks side by side so reviewers can see
the shift to a product-style output.
"""
from __future__ import annotations

import time

from community.schema import (
    CommunityAnalystReport,
    CommunitySentiment,
    CredibilityProfile,
    InsuranceFramework,
    SentimentProfile,
    TopicCluster,
    TrendProfile,
    UnifiedPost,
)
from financial_brief_formatter import (
    BriefingInput,
    MarketSnapshot,
    NewsItem,
    format_community_section,
    format_daily_brief,
)


def _post(platform: str, channel: str, title: str, engagement: int, author: str, age_hours: float):
    return UnifiedPost(
        platform=platform,
        post_id=f"{platform}-{channel}-{title[:10]}",
        channel=channel,
        title=title,
        engagement_raw=engagement,
        author=author,
        created_utc=time.time() - age_hours * 3600,
    )


def _make_cluster(
    cluster_id,
    headline,
    posts,
    sentiment_label,
    optimism,
    fear,
    uncertainty,
    skepticism,
    hype,
    discussion_focus,
    market_relevance,
    implications,
    triggers,
    credibility,
    trend,
):
    dims = {
        "optimism": optimism, "fear": fear, "uncertainty": uncertainty,
        "skepticism": skepticism, "hype": hype,
    }
    dom = max(dims.items(), key=lambda kv: kv[1])
    sent = SentimentProfile(
        label=sentiment_label,
        optimism=optimism, fear=fear, uncertainty=uncertainty,
        skepticism=skepticism, hype=hype,
        dominant_dimension=dom[0] if dom[1] > 0 else "",
        intensity=dom[1],
    )
    return TopicCluster(
        cluster_id=cluster_id,
        posts=posts,
        platforms=sorted({p.platform for p in posts}),
        rule_label="",
        heat_score=float(sum(p.engagement_raw for p in posts)),
        is_rising=trend.trend_direction == "rising",
        rise_ratio=2.0 if trend.trend_direction == "rising" else 1.0,
        headline=headline,
        discussion_focus=discussion_focus,
        market_relevance=market_relevance,
        insurance_framework=InsuranceFramework(
            implications=implications, triggers=triggers,
        ),
        insurance_angle=f"{implications} {triggers}".strip(),
        sentiment=sent,
        trend=trend,
        credibility=CredibilityProfile(overall=credibility, is_noise=credibility < 0.3),
        should_include_in_brief=True,
    )


def build_sample_report():
    # ── Topic 1: US Treasury safety premium eroded (Reddit) ────────────────
    t1_posts = [
        _post("reddit", "r/bonds", "IMF warns: US debt trajectory eroding the safety premium on treasuries", 340, "user1", 4),
        _post("reddit", "r/bonds", "Is the long end still a safe haven? discussion after IMF report", 210, "user2", 3),
        _post("reddit", "r/investing", "Debt dynamics and the term premium -- what changes", 180, "user3", 5),
        _post("reddit", "r/wallstreetbets", "TLT might finally be a falling-knife trade", 90, "user4", 6),
    ]
    t1 = _make_cluster(
        cluster_id="c1",
        headline="美债安全溢价被 IMF 警示抹去",
        posts=t1_posts,
        sentiment_label="mixed",
        optimism=0.15, fear=0.40, uncertainty=0.72, skepticism=0.35, hype=0.10,
        discussion_focus="多头认为溢价侵蚀被高估；空头担心财政路径已在定价中反映，长端利率上行风险加大",
        market_relevance="对长端久期利空，信用利差存在走阔风险；对银行净息差有边际影响",
        implications="对固收久期管理存在压力，长端利率若持续偏高会影响再投资收益率路径；对信用利差存在走阔的方向性压力",
        triggers="继续观察长端利率是否有效突破 4.45%，以及 IMF 叙事是否向 X/主流媒体扩散；只有在两者同时成立时，才考虑调整长端敞口",
        credibility=0.64,
        trend=TrendProfile(
            trend_direction="rising",
            persistence="continuing",
            platform_spread="reddit-led",
            discussion_breadth="moderate",
        ),
    )

    # ── Topic 2: Russia economy + Swedish warning (Reddit) ─────────────────
    t2_posts = [
        _post("reddit", "r/geopolitics", "Russia economy slowing despite oil windfall, Sweden warns", 120, "user5", 10),
        _post("reddit", "r/worldnews", "Swedish central bank cautious on oil-exporter knock-on risks", 60, "user6", 14),
    ]
    t2 = _make_cluster(
        cluster_id="c2",
        headline="俄罗斯经济放缓与瑞典预警",
        posts=t2_posts,
        sentiment_label="bearish",
        optimism=0.10, fear=0.60, uncertainty=0.45, skepticism=0.30, hype=0.10,
        discussion_focus="油价上涨是否真能支撑俄罗斯经济存在争议；瑞典的风险警示被用来质疑全球复苏的稳健度",
        market_relevance="对原油与能源资产存在承压；对整体风险偏好形成利空",
        implications="对能源板块敞口与通胀路径存在影响；对区域/新兴市场敞口构成边际风险",
        triggers="继续观察欧洲央行与瑞典央行后续表态；只有在油价与能源股同步回落时，才考虑调整能源敞口",
        credibility=0.55,
        trend=TrendProfile(
            trend_direction="stable",
            persistence="continuing",
            platform_spread="reddit-led",
            discussion_breadth="narrow",
        ),
    )

    # ── Topic 3: 10Y UST around 4.25 (Discord) ─────────────────────────────
    t3_posts = [
        _post("discord", "macro-traders", "10Y pinned around 4.25 — is this the top?", 55, "prof1", 2),
        _post("discord", "fixed-income", "FOMC minutes read mildly dovish, curve reaction muted", 45, "prof2", 3),
        _post("discord", "macro-traders", "What a break of 4.10 would do to REITs and growth", 40, "prof3", 5),
    ]
    t3 = _make_cluster(
        cluster_id="c3",
        headline="10Y UST 维持 4.25 附近，方向未决",
        posts=t3_posts,
        sentiment_label="mixed",
        optimism=0.35, fear=0.20, uncertainty=0.68, skepticism=0.25, hype=0.15,
        discussion_focus="一方认为收益率已见顶、降息预期在定价；另一方担心劳动力数据仍紧、区间继续震荡",
        market_relevance="利好久期下行空间，对金矿与利率敏感股存在利多；银行净息差承压",
        implications="对久期管理可能打开延长窗口；对再投资收益率的走向需结合后续利率路径确认；对利率敏感股/REITs 的定价存在方向性影响",
        triggers="继续观察 10Y 是否有效跌破 4.10% 或上破 4.45%；只有在伴随更软的通胀或劳动力数据时，才考虑真正调整久期敞口",
        credibility=0.88,
        trend=TrendProfile(
            trend_direction="rising",
            persistence="continuing",
            platform_spread="discord-led",
            discussion_breadth="moderate",
        ),
    )

    # ── Analyst report (report-level framework + bridge fields) ───────────
    report = CommunityAnalystReport(
        headline_topics=[t1, t2, t3],
        noise_topics=[],
        sentiment_structure="观望为主，不确定性较高；美债与利率话题上分歧与担忧并存，但尚未形成一致方向",
        cross_platform_signal="Reddit 聚焦财政/安全溢价风险，Discord 专业频道关注技术点位 4.25，两者在『利率路径』上形成互补而非冲突",
        insurance_framework=InsuranceFramework(
            implications=(
                "对固收久期与再投资收益率路径存在双向影响；对信用利差存在边际走阔压力；"
                "对利率敏感资产（REITs、成长股）的定价存在方向性风险"
            ),
            triggers=(
                "继续观察长端利率是否有效跌破 4.10% 或上破 4.45%，以及 IMF 安全溢价叙事是否向 X/主流媒体扩散；"
                "在任一条件成立前，暂不建议主动调整久期或信用敞口"
            ),
        ),
        insurance_angle="",  # intentionally empty so formatter uses framework
        brief_recommendation=(
            "突出 10Y 点位 + 美债安全溢价叙事的跨平台共振；弱化个股回报故事与短线情绪性讨论"
        ),
        platforms_covered=["reddit", "discord"],
        total_posts=149,
        total_clusters=5,
        platform_status=[
            "reddit=ok (62贴, 3簇)",
            "discord=ok (87条, 2簇)",
            "x=unavailable or not configured",
        ],
    )

    reddit_sent = CommunitySentiment(
        platform="reddit", trending_topics=[t1, t2],
        overall_sentiment="mixed", post_count=62, analyst_report=report,
    )
    discord_sent = CommunitySentiment(
        platform="discord", trending_topics=[t3],
        overall_sentiment="mixed", post_count=87, analyst_report=report,
    )

    return [reddit_sent, discord_sent], report


def build_sample_news():
    return [
        NewsItem(
            title="Texas Instruments ten-year return tops $3800 from $1000 — beats S&P and gold",
            summary="投资 1000 美元在 TXN 上十年回报 +290%，跑赢 S&P 500 与黄金。",
            source="Zacks",
            category="equity",
            why_it_matters="半导体数据中心需求+内部制造护城河支撑长期回报，但工业端疲软与成本上行是近期逆风。",
        ),
        NewsItem(
            title="JetBlue CEO rules out bankruptcy this year despite surging fuel costs amid Iran war",
            summary="捷蓝航空 CEO 否认年内破产可能，但承认燃料压力持续上行。",
            source="Reuters",
            category="equity",
            why_it_matters="航空业信用利差有走阔风险；若持有 JBLU 债建议评估 covenant。",
        ),
        NewsItem(
            title="Hormuz disruptions hit China's Christmas capital — holiday spending under pressure",
            summary="波斯湾航运受阻，塑料等原材料成本上行将传导至终端消费。",
            source="CNBC",
            category="macro",
            why_it_matters="全球消费 Q4 毛利承压；新兴市场外汇流动性风险上升。",
        ),
    ]


def main():
    sentiments, report = build_sample_report()
    news = build_sample_news()

    # Full brief
    briefing = BriefingInput(
        date_str="2026-04-21",
        market_snapshot=MarketSnapshot(
            us_equities="S&P 500 -0.24%，Nasdaq -0.26%，Dow -0.01%",
            rates="10Y Treasury yield 报 4.25%，较前一交易日持平",
            asia_sg="STI +0.08%",
        ),
        news_items=news,
        watchlist=[
            "关注今晚是否有新的关键宏观数据",
            "关注美联储官员表态",
            "关注长端利率是否继续波动",
        ],
        community_sentiments=sentiments,
        community_report=report,
    )
    # Don't pass llm_callable — news enricher will fall back to heuristics.
    text = format_daily_brief(briefing, llm_callable=None)
    print(text)


if __name__ == "__main__":
    main()
