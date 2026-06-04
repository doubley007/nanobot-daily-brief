"""
Session Memory — 短期会话记忆（v6 升级：评分式 follow-up resolver）。

目标：支持 follow-up 对话，让 bot 知道"上一个问题是关于什么的"，
而不是每句话都当成全新的独立问题。

v6 升级：
  - Follow-up resolver 升级为基于规则权重评分（0.0–1.0）
  - 支持更模糊的问法："那个怎么样？"、"这个行吗？"、"还能追吗？"
  - SessionContext 增加 resolved_from_session + resolver_confidence
  - 默认持久化到 logs/session_memory.json

SessionTurn 字段：
  asset           当时讨论的资产
  intent          当时的意图（market_decision / market_summary / emotional_chat）
  emotion         用户情绪
  action          决策引擎输出的 action（buy_consider / hold / avoid / unknown）
  topic           本轮的主题摘要（一句话）
  ts              时间戳
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque

logger = logging.getLogger(__name__)

SESSION_MAX_TURNS = 10
SESSION_TTL_SECONDS = 3600  # 1 hour — older turns ignored for follow-up resolution


# ─── Follow-up resolver (v6: scored rules) ────────────────────────────────────

# High-confidence follow-up openers (score 0.9)
_HIGH_CONF_PHRASES = [
    "那现在", "那如果", "那这个", "那个怎么样", "那个还行吗",
    "那个能买吗", "如果我已经买了", "如果我买了", "如果已经持有",
    "那还能追吗", "还能再追吗", "还能追进去吗",
    "then what", "what about then", "if i already",
]

# Medium-confidence follow-up signals (score 0.7)
_MED_CONF_PHRASES = [
    "那", "然后", "那么", "那如果", "还能", "那比", "还有",
    "之后", "如果我", "如果已经", "that means", "so then",
    "what about", "and if", "so now", "follow up",
    "another question", "one more", "related",
]

# Demonstrative pronouns — high-conf only when previous turn exists (score 0.85)
_DEMONSTRATIVE_PHRASES = [
    "这个", "那个", "这只", "那只", "这支", "那支",
    "this one", "that one", "it",
]

# Asset switch patterns — user explicitly mentions a new asset starting with "那"
_ASSET_SWITCH_STARTERS = [
    "那比特币", "那黄金", "那英伟达", "那sp500", "那石油",
    "那原油", "那美元", "那特斯拉",
    "比特币呢", "黄金呢", "英伟达呢", "比特币怎么样", "黄金怎么样",
    "what about bitcoin", "what about gold", "what about nvidia",
]


def _score_followup(text: str, has_prior_session: bool) -> float:
    """
    Return a follow-up confidence score in [0, 1].
    0 = definitely a new question
    1 = definitely continuing from previous turn

    Decision boundary: score >= 0.65 → treat as follow-up.
    """
    lower = text.lower().strip()
    if not lower:
        return 0.0

    # Explicit asset switch starters checked first — takes priority over short-message heuristic
    # We return 0.5 to signal "switch" — caller handles this case separately
    for p in _ASSET_SWITCH_STARTERS:
        if lower.startswith(p) or p in lower[:20]:
            return 0.5  # special: asset switch

    # Very short messages almost always refer to prior context
    if len(lower) <= 6:
        return 0.9 if has_prior_session else 0.0

    # High-confidence openers
    for p in _HIGH_CONF_PHRASES:
        if lower.startswith(p) or lower[:20].__contains__(p):
            return 0.9

    # Demonstrative pronouns — only meaningful with prior session
    for p in _DEMONSTRATIVE_PHRASES:
        if p in lower[:20]:
            return 0.85 if has_prior_session else 0.2

    # Medium-confidence markers
    score = 0.0
    hits = sum(1 for p in _MED_CONF_PHRASES if p in lower)
    if hits >= 2:
        score = 0.75
    elif hits == 1:
        score = 0.65

    # No follow-up markers at all
    return score


def _looks_like_followup(text: str) -> bool:
    """Legacy compat: returns True if score >= 0.65 with any prior context."""
    return _score_followup(text, has_prior_session=True) >= 0.65


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class SessionTurn:
    asset: str | None
    intent: str      # "market_decision" | "market_summary" | "emotional_chat"
    emotion: str     # primary emotion label
    action: str      # "buy_consider" | "hold" | "avoid" | "unknown"
    topic: str       # one-line topic summary
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionTurn":
        return cls(
            asset=d.get("asset"),
            intent=d.get("intent", "emotional_chat"),
            emotion=d.get("emotion", "neutral"),
            action=d.get("action", "unknown"),
            topic=d.get("topic", ""),
            ts=float(d.get("ts", 0)),
        )


@dataclass
class SessionContext:
    """Resolved session context for the current turn (v6: enhanced)."""
    resolved_asset: str | None      # possibly inherited from previous turn
    recent_action: str | None       # last decision action
    recent_emotion: str | None      # last emotion
    is_followup: bool               # did we inherit context from previous turn?
    prev_topic: str | None          # what was discussed last
    # v6 additions
    resolved_from_session: bool = False   # True if asset came from session (not text)
    resolver_confidence: float = 0.0      # 0.0–1.0 follow-up confidence score


# ─── Per-user session store ───────────────────────────────────────────────────

class UserSession:
    """Ring buffer of the last N turns for one user."""

    def __init__(self, max_turns: int = SESSION_MAX_TURNS) -> None:
        self._turns: Deque[SessionTurn] = deque(maxlen=max_turns)

    def push(self, turn: SessionTurn) -> None:
        self._turns.append(turn)

    def recent_turns(self, max_age_seconds: float = SESSION_TTL_SECONDS) -> list[SessionTurn]:
        cutoff = time.time() - max_age_seconds
        return [t for t in self._turns if t.ts >= cutoff]

    def last_turn(self) -> SessionTurn | None:
        turns = self.recent_turns()
        return turns[-1] if turns else None

    def last_asset(self) -> str | None:
        for t in reversed(self.recent_turns()):
            if t.asset:
                return t.asset
        return None

    def last_action(self) -> str | None:
        t = self.last_turn()
        return t.action if t else None

    def resolve_context(self, text: str, detected_asset: str | None) -> SessionContext:
        """
        v6: Scored follow-up resolver.

        Scoring logic:
          score >= 0.65 → follow-up (inherit session asset if none detected)
          score == 0.5  → asset switch (new asset explicitly named, keep intent from session)
          score < 0.65  → new independent question

        If detected_asset is already set, we never override it (user was explicit).
        We still record the score for trace visibility.
        """
        last = self.last_turn()
        has_prior = last is not None

        confidence = _score_followup(text, has_prior_session=has_prior)

        is_followup = False
        resolved_from_session = False
        resolved_asset = detected_asset

        if detected_asset is None and has_prior:
            if confidence >= 0.65:
                resolved_asset = last.asset  # type: ignore[union-attr]
                is_followup = True
                resolved_from_session = True
            # score == 0.5 means asset switch — detected_asset should have caught it
            # so we just leave resolved_asset = None (route_query will handle fallback)
        elif detected_asset is not None and confidence >= 0.65 and has_prior:
            # User mentioned a new asset AND it looks like a follow-up
            # (e.g., "那比特币呢？" — switch to bitcoin, inherit session intent)
            is_followup = True
            resolved_from_session = False  # asset came from text

        return SessionContext(
            resolved_asset=resolved_asset,
            recent_action=last.action if last else None,
            recent_emotion=last.emotion if last else None,
            is_followup=is_followup,
            prev_topic=last.topic if last else None,
            resolved_from_session=resolved_from_session,
            resolver_confidence=round(confidence, 2),
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._turns]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]],
                  max_turns: int = SESSION_MAX_TURNS) -> "UserSession":
        s = cls(max_turns=max_turns)
        for d in data:
            s._turns.append(SessionTurn.from_dict(d))
        return s


# ─── Global session registry ──────────────────────────────────────────────────

# Max seconds since last activity before a user's session is pruned on save
_SESSION_USER_TTL = 86400 * 7  # 7 days


def _resolve_session_path() -> Path:
    """
    v6: default persistence enabled.
    Override with SESSION_MEMORY_FILE env var (e.g., for tests).
    Set SESSION_MEMORY_FILE='' (empty string) to disable persistence.
    """
    env = os.getenv("SESSION_MEMORY_FILE", None)
    if env is not None:
        # Explicitly set (possibly empty) — respect it
        return Path(env) if env.strip() else Path("/dev/null")
    # Default: persist to logs/session_memory.json
    return Path(__file__).resolve().parents[2] / "logs" / "session_memory.json"


class SessionMemoryStore:
    """
    Registry of all user sessions. v6: persisted by default.

    Persistence path (in priority order):
      1. `path` argument (explicit, e.g. from tests)
      2. SESSION_MEMORY_FILE env var (override or disable with empty string)
      3. Default: logs/session_memory.json

    Set SESSION_MEMORY_FILE='' to disable persistence (pure in-memory).
    """

    def __init__(self, path: str | Path | None = None,
                 max_turns: int = SESSION_MAX_TURNS) -> None:
        if path is not None:
            self._path: Path | None = Path(path) if str(path) else None
        else:
            resolved = _resolve_session_path()
            # /dev/null means "disable persistence"
            self._path = None if str(resolved) == "/dev/null" else resolved
        self._max_turns = max_turns
        self._sessions: dict[str, UserSession] = {}
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            for uid, turns_data in raw.get("sessions", {}).items():
                self._sessions[uid] = UserSession.from_list(
                    turns_data, max_turns=self._max_turns)
            logger.info("SessionMemory: loaded %d users from %s",
                        len(self._sessions), self._path)
        except Exception as e:
            logger.warning("SessionMemory: load failed: %s", e)

    def _prune_inactive_users(self) -> None:
        """Remove users whose last turn is older than _SESSION_USER_TTL."""
        cutoff = time.time() - _SESSION_USER_TTL
        stale = [
            uid for uid, s in self._sessions.items()
            if not any(t.ts >= cutoff for t in s._turns)
        ]
        for uid in stale:
            del self._sessions[uid]
        if stale:
            logger.debug("SessionMemory: pruned %d inactive users", len(stale))

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._prune_inactive_users()
            payload = {
                "sessions": {uid: s.to_list() for uid, s in self._sessions.items()},
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self._path)
        except Exception as e:
            logger.warning("SessionMemory: save failed: %s", e)

    def _get_or_create(self, user_id: str) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(self._max_turns)
        return self._sessions[user_id]

    def push(self, user_id: str | int, turn: SessionTurn) -> None:
        uid = str(user_id)
        self._get_or_create(uid).push(turn)
        self._save()

    def resolve_context(
        self,
        user_id: str | int,
        text: str,
        detected_asset: str | None,
    ) -> SessionContext:
        uid = str(user_id)
        session = self._sessions.get(uid, UserSession(self._max_turns))
        return session.resolve_context(text, detected_asset)

    def get_session(self, user_id: str | int) -> UserSession:
        return self._get_or_create(str(user_id))

    def clear_user(self, user_id: str | int) -> None:
        uid = str(user_id)
        if uid in self._sessions:
            del self._sessions[uid]
            self._save()


# ─── Module singleton ─────────────────────────────────────────────────────────

_default_session_store: SessionMemoryStore | None = None


def default_session_store() -> SessionMemoryStore:
    global _default_session_store
    if _default_session_store is None:
        _default_session_store = SessionMemoryStore()
    return _default_session_store


def reset_session_store() -> None:
    global _default_session_store
    _default_session_store = None


# ─── Pipeline helpers ─────────────────────────────────────────────────────────

def record_turn(
    user_id: str | int | None,
    asset: str | None,
    intent: str,
    emotion: str,
    action: str,
    topic: str,
) -> None:
    """Record a completed turn into session memory. Safe to call; never raises."""
    if user_id is None:
        return
    try:
        turn = SessionTurn(
            asset=asset, intent=intent, emotion=emotion,
            action=action, topic=topic,
        )
        default_session_store().push(user_id, turn)
    except Exception as e:
        logger.debug("session_memory.record_turn failed (non-fatal): %s", e)


def resolve_session_context(
    user_id: str | int | None,
    text: str,
    detected_asset: str | None,
) -> SessionContext:
    """
    Resolve session context for this turn.
    If user_id is None, returns a blank context with detected_asset only.
    """
    if user_id is None:
        return SessionContext(
            resolved_asset=detected_asset,
            recent_action=None,
            recent_emotion=None,
            is_followup=False,
            prev_topic=None,
        )
    return default_session_store().resolve_context(user_id, text, detected_asset)
