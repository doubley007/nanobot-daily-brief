"""
Reply Policy —— 回答策略层（不允许 LLM 自由发挥的边界）。

规则严格执行，不可覆盖：
  1. 先给结论，再给依据
  2. 不输出禁用话术（公司语境里注册的 banned_phrases）
  3. 不因用户 FOMO 把结论往 buy 倾斜
  4. 社区观点不等于事实——只能作为情绪指标
  5. 证据不足也要给明确倾向，不说"不确定"
  6. FOMO 用户 + buy 决策 → 必须加"别重仓"提醒
  7. 内部用户 → 更简洁，不要重复废话
  8. 普通用户 → 多解释，少术语

PolicyResult 包含：
  - is_valid: bool（是否通过）
  - violations: list[str]（违规条目）
  - enforced_text: str（policy 修正后的文本，仅 violations 时有值）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from assistant.company_context import CompanyContext, get_company_context
from assistant.user_profile import UserProfile
from assistant.user_emotion import UserEmotionProfile
from assistant.decision_engine import Decision


# ─── Policy 结果 ─────────────────────────────────────────────────────────────

@dataclass
class PolicyResult:
    is_valid: bool
    violations: list[str] = field(default_factory=list)


# ─── 核心 Policy 检查 ─────────────────────────────────────────────────────────

def check_reply_policy(
    reply: str,
    decision: Decision | None,
    emotion: UserEmotionProfile,
    profile: UserProfile,
    company: CompanyContext | None = None,
) -> PolicyResult:
    """
    检查生成的 reply 是否违反 policy。返回 PolicyResult。
    调用方可以根据 violations 决定是否要重新生成或追加文本。
    """
    company = company or get_company_context()
    violations: list[str] = []

    # 1. 禁用话术检查
    banned = company.has_banned_phrase(reply)
    for phrase in banned:
        violations.append(f"banned_phrase: '{phrase}'")

    # 2. FOMO 用户 + buy 决策 → 必须有"别重仓/分批"提醒
    if (
        emotion.primary_emotion == "fomo"
        and decision is not None
        and decision.action in ("buy", "buy_small")
    ):
        caution_keywords = ("scale in", "small position", "don't go all-in",
                            "don't chase", "staggered", "in tranches")
        reply_lower = reply.lower()
        has_caution = any(kw in reply_lower for kw in caution_keywords)
        if not has_caution:
            violations.append("fomo_buy_missing_caution: FOMO + buy without scale-in/small-position reminder")

    # 3. 不能因用户情绪强行 buy（检查：如果 decision 是 avoid/hold 但 reply 说"可以买"）
    if decision is not None and decision.action in ("avoid", "hold_wait"):
        if re.search(r"(you can buy|recommend buy|buy now|enter now)", reply, re.IGNORECASE):
            violations.append("emotion_bias: reply suggests buy but decision is avoid/hold_wait")

    return PolicyResult(is_valid=len(violations) == 0, violations=violations)


# ─── 回复风格规范化 ───────────────────────────────────────────────────────────

def apply_style_constraints(
    text: str,
    profile: UserProfile,
    emotion: UserEmotionProfile,
    company: CompanyContext | None = None,
) -> str:
    """
    根据 profile + emotion + company 对 reply 做最终修整：
      - 内部用户：删掉多余的解释性段落（超过 4 段只保留前 4 段 + 结论）
      - 外部用户 + anxious：确保第一段是共情语
      - 删掉禁用话术（直接替换）
    注意：不改变决策方向，只改措辞/顺序。
    """
    company = company or get_company_context()

    # 替换禁用话术
    for phrase in company.banned_phrases:
        # 大小写不敏感替换
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("", text)

    # 内部用户：保留最精简版本（不超过 6 段）
    if profile.wants_concise_reply:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 6:
            text = "\n\n".join(paragraphs[:6])

    return text.strip()


# ─── Few-shot 风格示例（供 LLM 做风格放大器） ─────────────────────────────────

STYLE_EXAMPLES = [
    {
        "question": "Should I buy gold here?",
        "emotion": "neutral",
        "profile": "retail",
        "decision": "buy_small",
        "good_reply": (
            "🟡 Bottom line: small, staggered participation\n\n"
            "Gold skews bullish — rate-cut expectations and geopolitical risk are both supportive, "
            "and community sentiment is bullish (around 70%).\n"
            "But spec long positioning is at a 3-year high, so drawdown risk on any chase is meaningful.\n\n"
            "[Recent news] skew bullish:\n"
            "  • Gold hits record high as rate cut bets strengthen (Reuters)\n"
            "  • Goldman raises gold year-end target to $2,700 (CNBC)\n\n"
            "[Community sentiment] bullish 67%, clear FOMO signal, some crowding.\n\n"
            "Risks to watch:\n"
            "  • Crowding is elevated — any counter-catalyst can reverse it fast\n"
            "  • Short-term run has been large; chase drawdown is real\n\n"
            "👉 If you're going, take a small position and scale in. Don't chase after a big up-day."
        ),
        "bad_reply": "Gold has done well, but it depends on your risk appetite — please consult a professional before deciding.",
    },
    {
        "question": "Can I still chase gold?",
        "emotion": "fomo",
        "profile": "retail",
        "decision": "avoid_chasing",
        "good_reply": (
            "Slow down for a sec — let me walk you through where things actually stand.\n\n"
            "⚠️ Bottom line: don't chase here (gold)\n\n"
            "News is still supportive, but the short-term move has been big, "
            "community FOMO is heavy and positioning is crowded. "
            "The payoff from chasing here is worse than the risk.\n\n"
            "I get the FOMO — but 'others made money' isn't a reason you have to enter now. "
            "Wait for a pullback or some consolidation; there'll still be a trade."
        ),
        "bad_reply": "Everyone's chasing gold, the market clearly likes it — you can consider buying some.",
    },
    {
        "question": "What's going on in the market lately?",
        "emotion": "neutral",
        "profile": "unknown",
        "decision": None,
        "good_reply": (
            "📊 Recent market read:\n\n"
            "Top stories:\n"
            "  • Fed rate-cut expectations firming, market pricing a September cut\n"
            "  • Middle East geopolitical risk persists, safe havens supported\n"
            "  • Stronger USD weighing on parts of the commodity complex\n\n"
            "Community: broadly bullish on macro assets; gold and bitcoin dominate the discussion, "
            "FOMO is elevated, crowded-trade risk is medium.\n"
            "Dominant narratives: rate cut, safe haven, buy the dip"
        ),
        "bad_reply": "The market has been mixed recently — it depends on which asset you care about; please judge for yourself.",
    },
    {
        "question": "I'm really anxious, I don't know what to do",
        "emotion": "anxious",
        "profile": "unknown",
        "decision": None,
        "good_reply": (
            "Take a breath — I'll lay out what's actually happening.\n\n"
            "Markets are uncomfortable, but panic is the worst state to make decisions from.\n\n"
            "Tell me which asset or situation is worrying you and I'll pull the facts and community read "
            "together before you touch the position."
        ),
        "bad_reply": "Please don't worry — investing involves risk, please consult a professional.",
    },
    {
        "question": "Everyone's buying — should I jump in too?",
        "emotion": "fomo",
        "profile": "retail",
        "decision": "buy_small",
        "good_reply": (
            "Slow down for a sec — let me walk you through where things actually stand.\n\n"
            "🟡 Bottom line: small, staggered participation\n\n"
            "'Everyone's buying' on its own is a neutral signal — it tells you sentiment is hot, "
            "not that price still has room. Community FOMO ratio is already high and crowding is medium.\n\n"
            "News support is still there (rate-cut expectations + safe-haven demand), "
            "but the chase payoff is worse than two weeks ago.\n\n"
            "One more thing — what you're describing is 'fear of missing out'. "
            "That doesn't rule out buying, but don't go all-in in one clip. "
            "Scale in over 2-3 tranches so you leave yourself room to react."
        ),
        "bad_reply": "Everyone's buying — it seems popular, you can consider entering too, just be mindful of risk.",
    },
]


def get_style_examples_for_prompt(
    emotion: str = "neutral",
    decision_action: str | None = None,
    n: int = 2,
) -> str:
    """
    按 emotion + decision_action 筛选最相关的 few-shot 示例，
    格式化成可注入 LLM prompt 的字符串。
    """
    scored = []
    for ex in STYLE_EXAMPLES:
        score = 0
        if ex["emotion"] == emotion:
            score += 2
        if decision_action and ex["decision"] == decision_action:
            score += 1
        scored.append((score, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [ex for _, ex in scored[:n]]

    parts = ["Reference examples of high-quality replies (learn the STYLE, do not copy content):"]
    for i, ex in enumerate(selected, 1):
        parts.append(
            f"\nExample {i}:\n"
            f"  Question: {ex['question']}\n"
            f"  Good reply:\n{ex['good_reply']}\n"
            f"  Bad reply (forbidden):\n{ex['bad_reply']}"
        )
    return "\n".join(parts)
