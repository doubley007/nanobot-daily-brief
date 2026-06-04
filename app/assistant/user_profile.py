"""
User Profile System —— 用户画像，支持持久化。

持久化方案：
  - 数据写入 USER_PROFILES_FILE 指向的 JSON 文件（或项目根 user_profiles.json）
  - dirty flag：任何 update_from_interaction() 触发实际变更时设置
  - flush() 把 dirty profiles 写回文件（pipeline 调用，或进程退出时调用）
  - 文件格式包含 schema_version 字段，便于未来演进
  - fallback profile（DEFAULT_PROFILE）永远不写盘

加载顺序：
  1. USER_PROFILES_FILE 环境变量 → JSON 文件
  2. 项目根 user_profiles.json
  3. DEFAULT_PROFILE fallback
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PROFILE_SCHEMA_VERSION = 2  # bump when fields change


# ─── 枚举 ────────────────────────────────────────────────────────────────────

Role = Literal["insider", "pm", "analyst", "trader", "retail", "unknown"]
RiskPreference = Literal["aggressive", "moderate", "conservative", "unknown"]
PreferredStyle = Literal["concise", "analytical", "educational", "unknown"]


# ─── 数据类 ──────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str                              # Telegram user_id (as str) or any identifier
    display_name: str = "User"
    role: Role = "unknown"
    risk_preference: RiskPreference = "unknown"
    preferred_style: PreferredStyle = "analytical"
    interests: list[str] = field(default_factory=list)     # e.g. ["gold", "crypto", "macro"]
    behavior_traits: list[str] = field(default_factory=list)  # e.g. ["fomo_prone"]
    language: str = "zh"
    is_internal: bool = False
    last_updated_at: float = 0.0              # unix timestamp of last learned update

    # ── 行为派生属性 ─────────────────────────────────────────────────────────

    @property
    def needs_simplified_language(self) -> bool:
        return self.role in ("retail", "unknown") and not self.is_internal

    @property
    def wants_concise_reply(self) -> bool:
        return self.role in ("insider", "pm", "trader") or self.is_internal

    @property
    def fomo_prone(self) -> bool:
        return "fomo_prone" in self.behavior_traits

    def to_context_block(self) -> str:
        """生成注入 prompt 的用户画像段。"""
        lines = [
            f"[用户画像]",
            f"角色：{self.role}，偏好风格：{self.preferred_style}",
            f"风险偏好：{self.risk_preference}",
        ]
        if self.interests:
            lines.append(f"关注资产/主题：{', '.join(self.interests[:5])}")
        if self.behavior_traits:
            lines.append(f"行为特征：{', '.join(self.behavior_traits[:3])}")
        if self.is_internal:
            lines.append("身份：内部用户，回答要更简洁、更结论导向")
        elif self.needs_simplified_language:
            lines.append("身份：普通投资者，需要更多解释，少用专业术语")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_saveable_dict(self) -> dict[str, Any]:
        """Serializable dict for JSON persistence — only learned/mutable fields."""
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role,
            "risk_preference": self.risk_preference,
            "preferred_style": self.preferred_style,
            "interests": list(self.interests),
            "behavior_traits": list(self.behavior_traits),
            "language": self.language,
            "is_internal": self.is_internal,
            "last_updated_at": self.last_updated_at,
        }


# ─── 默认 profile ─────────────────────────────────────────────────────────────

DEFAULT_PROFILE = UserProfile(
    user_id="default",
    display_name="用户",
    role="unknown",
    risk_preference="moderate",
    preferred_style="analytical",
    interests=[],
    behavior_traits=[],
    language="zh",
    is_internal=False,
)


# ─── Profile 仓库 ─────────────────────────────────────────────────────────────

class UserProfileStore:
    """
    内存 dict 存储 + JSON 文件持久化。
    - load from JSON at first access
    - dirty flag: set when learned traits change
    - flush() writes dirty profiles back to the same file
    - DEFAULT_PROFILE is never written to disk
    """

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}
        self._loaded = False
        self._loaded_path: Path | None = None     # file we loaded from (= file we save to)
        self._dirty_ids: set[str] = set()         # user_ids that need flushing
        self._interaction_counters: dict[str, dict[str, int]] = {}

    # ── Conservative thresholds ───────────────────────────────────────────────

    _FOMO_THRESHOLD = 3
    _INTEREST_THRESHOLD = 2
    _CONCISE_THRESHOLD = 3

    # ── Loading ───────────────────────────────────────────────────────────────

    def _resolve_path(self) -> Path | None:
        env_path = os.getenv("USER_PROFILES_FILE", "").strip()
        if env_path:
            return Path(env_path)
        default_file = Path(__file__).resolve().parents[2] / "user_profiles.json"
        return default_file

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        path = self._resolve_path()
        if path is None:
            return
        self._loaded_path = path

        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("profiles", []):
                uid = str(entry.get("user_id", ""))
                if not uid:
                    continue
                p = UserProfile(
                    user_id=uid,
                    display_name=entry.get("display_name", "User"),
                    role=entry.get("role", "unknown"),
                    risk_preference=entry.get("risk_preference", "unknown"),
                    preferred_style=entry.get("preferred_style", "analytical"),
                    interests=list(entry.get("interests", [])),
                    behavior_traits=list(entry.get("behavior_traits", [])),
                    language=entry.get("language", "zh"),
                    is_internal=bool(entry.get("is_internal", False)),
                    last_updated_at=float(entry.get("last_updated_at", 0.0)),
                )
                self._profiles[uid] = p
            logger.info("UserProfileStore: loaded %d profiles from %s",
                        len(self._profiles), path)
        except Exception as e:
            logger.warning("UserProfileStore: failed to load %s: %s", path, e)

    # ── Persistence ───────────────────────────────────────────────────────────

    def flush(self) -> bool:
        """
        Write dirty profiles back to the JSON file.
        Returns True if anything was written, False otherwise.
        Only profiles that actually changed are considered dirty.
        DEFAULT_PROFILE is never saved.
        """
        if not self._dirty_ids:
            return False

        self._ensure_loaded()
        save_path = self._loaded_path or self._resolve_path()
        if save_path is None:
            return False

        # Merge: start from what's on disk (if exists), overlay current state
        existing_profiles: dict[str, dict] = {}
        if save_path.exists():
            try:
                data = json.loads(save_path.read_text(encoding="utf-8"))
                for entry in data.get("profiles", []):
                    uid = str(entry.get("user_id", ""))
                    if uid:
                        existing_profiles[uid] = entry
            except Exception as e:
                logger.warning("UserProfileStore.flush: could not read existing file: %s", e)

        for uid in self._dirty_ids:
            if uid == "default":
                continue
            p = self._profiles.get(uid)
            if p is None:
                continue
            existing_profiles[uid] = p.to_saveable_dict()

        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profiles": list(existing_profiles.values()),
        }

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = save_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(save_path)  # atomic rename
            self._dirty_ids.clear()
            logger.info("UserProfileStore: flushed %d profile(s) to %s",
                        len(payload["profiles"]), save_path)
            return True
        except Exception as e:
            logger.warning("UserProfileStore.flush failed: %s", e)
            return False

    def save(self, user_id: str | int | None = None) -> bool:
        """Force-save a specific profile (or all if user_id is None)."""
        self._ensure_loaded()
        if user_id is not None:
            uid = str(user_id)
            if uid in self._profiles:
                self._dirty_ids.add(uid)
        else:
            self._dirty_ids.update(self._profiles.keys())
        return self.flush()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def get(self, user_id: str | int | None) -> UserProfile:
        self._ensure_loaded()
        if user_id is None:
            return DEFAULT_PROFILE
        return self._profiles.get(str(user_id), DEFAULT_PROFILE)

    def set(self, profile: UserProfile) -> None:
        """Register (or overwrite) a profile in memory. Does not auto-save."""
        self._profiles[profile.user_id] = profile
        self._loaded = True

    def reload(self) -> None:
        """Test hook: clear cache and counters, re-load from file on next access."""
        self._profiles.clear()
        self._loaded = False
        self._loaded_path = None
        self._dirty_ids.clear()
        self._interaction_counters.clear()

    # ── Interaction-based learning ────────────────────────────────────────────

    def update_from_interaction(
        self,
        user_id: str | int | None,
        interaction_type: str,
        asset: str | None = None,
    ) -> bool:
        """
        Record one interaction and update UserProfile traits when thresholds are met.

        interaction_type values:
          "fomo"             — user message showed FOMO emotion
          "asset_mention"    — user asked about a specific asset
          "concise_feedback" — user prefers concise answers

        Returns True if any trait was updated (triggers dirty flag + deferred flush).
        """
        if user_id is None:
            return False
        uid = str(user_id)
        self._ensure_loaded()

        profile = self._profiles.get(uid)
        if profile is None:
            return False

        if uid not in self._interaction_counters:
            self._interaction_counters[uid] = {}
        counters = self._interaction_counters[uid]

        updated = False

        if interaction_type == "fomo":
            counters["fomo"] = counters.get("fomo", 0) + 1
            if (counters["fomo"] >= self._FOMO_THRESHOLD
                    and "fomo_prone" not in profile.behavior_traits):
                profile.behavior_traits = profile.behavior_traits + ["fomo_prone"]
                updated = True
                logger.info("user_profile: %s → fomo_prone added", uid)

        elif interaction_type == "asset_mention" and asset:
            asset_key = f"asset:{asset}"
            counters[asset_key] = counters.get(asset_key, 0) + 1
            if (counters[asset_key] >= self._INTEREST_THRESHOLD
                    and asset not in profile.interests):
                profile.interests = profile.interests + [asset]
                updated = True
                logger.info("user_profile: %s → interest '%s' added", uid, asset)

        elif interaction_type == "concise_feedback":
            counters["concise"] = counters.get("concise", 0) + 1
            if (counters["concise"] >= self._CONCISE_THRESHOLD
                    and profile.preferred_style != "concise"):
                profile.preferred_style = "concise"
                updated = True
                logger.info("user_profile: %s → preferred_style → concise", uid)

        if updated:
            profile.last_updated_at = time.time()
            self._dirty_ids.add(uid)
            # Non-blocking deferred flush: caller can call flush() explicitly,
            # or it happens automatically via pipeline's periodic flush call.

        return updated


# ─── 默认单例 ─────────────────────────────────────────────────────────────────

_default_store: UserProfileStore | None = None


def get_profile_store() -> UserProfileStore:
    global _default_store
    if _default_store is None:
        _default_store = UserProfileStore()
    return _default_store


def get_user_profile(user_id: str | int | None) -> UserProfile:
    return get_profile_store().get(user_id)


def update_profile_from_interaction(
    user_id: str | int | None,
    interaction_type: str,
    asset: str | None = None,
) -> bool:
    """Convenience wrapper — update the default store's profile for user_id."""
    return get_profile_store().update_from_interaction(user_id, interaction_type, asset)


def flush_profile_store() -> bool:
    """Flush dirty profiles to disk. Call from pipeline after each answered Q."""
    return get_profile_store().flush()


def reset_profile_store() -> None:
    """Test hook."""
    global _default_store
    _default_store = None
