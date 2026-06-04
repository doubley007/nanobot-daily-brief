"""
Holdings / Portfolio Module — 用户持仓感知（v5 增强：成本价与盈亏语境）。

v5 新增：
  - avg_cost 字段现在主动用于计算盈亏状态 (pnl_status)
  - pnl_status: "in_profit" | "underwater" | "near_cost" | "unknown"
  - to_context_block() 注入盈亏语境
  - holdings_reply_addendum() 区分盈利/亏损/接近成本价的措辞
  - /setholding 命令扩展：/setholding gold medium 3320 long

持仓感知只影响回复措辞/上下文，不直接改变市场决策 action。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PositionSize = Literal["none", "small", "medium", "large"]
Horizon = Literal["short", "mid", "long", "unknown"]

HOLDINGS_SCHEMA_VERSION = 1


@dataclass
class Holding:
    user_id: str
    asset: str
    position_size: PositionSize = "none"   # none | small | medium | large
    avg_cost: float | None = None          # optional cost basis
    horizon: Horizon = "unknown"           # short | mid | long
    notes: str = ""
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pnl_status(self, current_price: float | None = None) -> str:
        """
        Returns: "in_profit" | "underwater" | "near_cost" | "unknown"
        near_cost = within ±3% of avg_cost.
        Requires current_price to be meaningful; falls back to "unknown".
        """
        if self.avg_cost is None or self.avg_cost <= 0:
            return "unknown"
        if current_price is None or current_price <= 0:
            return "unknown"
        pct = (current_price - self.avg_cost) / self.avg_cost
        if abs(pct) <= 0.03:
            return "near_cost"
        return "in_profit" if pct > 0 else "underwater"

    def to_context_block(self, current_price: float | None = None) -> str:
        if self.position_size == "none":
            return f"[Holding] {self.asset}: no position"
        lines = [f"[Holding] {self.asset}: size={self.position_size}, horizon={self.horizon}"]
        if self.avg_cost is not None:
            cost_str = f"{self.avg_cost:,.2f}"
            pnl = self.pnl_status(current_price)
            pnl_label = {
                "in_profit": "in profit",
                "underwater": "underwater",
                "near_cost": "near cost",
                "unknown": "",
            }.get(pnl, "")
            lines.append(f"  Cost: {cost_str}" + (f" ({pnl_label})" if pnl_label else ""))
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)

    @property
    def has_position(self) -> bool:
        return self.position_size != "none"

    @property
    def is_heavy(self) -> bool:
        return self.position_size == "large"


# ─── 存储 ────────────────────────────────────────────────────────────────────

def _resolve_holdings_path() -> Path:
    import os
    env = os.getenv("HOLDINGS_FILE", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "holdings.json"


class HoldingsStore:
    """
    In-memory dict backed by JSON file.
    Key: (user_id, asset) → Holding.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _resolve_holdings_path()
        self._data: dict[str, dict[str, Holding]] = {}  # {user_id: {asset: Holding}}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for entry in raw.get("holdings", []):
                uid = str(entry.get("user_id", ""))
                asset = str(entry.get("asset", ""))
                if not uid or not asset:
                    continue
                h = Holding(
                    user_id=uid, asset=asset,
                    position_size=entry.get("position_size", "none"),
                    avg_cost=entry.get("avg_cost"),
                    horizon=entry.get("horizon", "unknown"),
                    notes=entry.get("notes", ""),
                    updated_at=float(entry.get("updated_at", 0)),
                )
                self._data.setdefault(uid, {})[asset] = h
            logger.info("HoldingsStore: loaded from %s", self._path)
        except Exception as e:
            logger.warning("HoldingsStore: load failed: %s", e)

    def _save(self) -> None:
        try:
            all_holdings = [
                h.to_dict()
                for user_holdings in self._data.values()
                for h in user_holdings.values()
            ]
            payload = {
                "schema_version": HOLDINGS_SCHEMA_VERSION,
                "holdings": all_holdings,
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self._path)
        except Exception as e:
            logger.warning("HoldingsStore: save failed: %s", e)

    def get(self, user_id: str | int | None, asset: str) -> Holding | None:
        """Return holding for (user_id, asset), or None if not set."""
        self._ensure_loaded()
        if user_id is None:
            return None
        uid = str(user_id)
        return self._data.get(uid, {}).get(asset)

    def get_all(self, user_id: str | int | None) -> list[Holding]:
        """Return all holdings for a user."""
        self._ensure_loaded()
        if user_id is None:
            return []
        uid = str(user_id)
        return list(self._data.get(uid, {}).values())

    def set(
        self,
        user_id: str | int,
        asset: str,
        position_size: PositionSize = "small",
        avg_cost: float | None = None,
        horizon: Horizon = "unknown",
        notes: str = "",
    ) -> Holding:
        self._ensure_loaded()
        uid = str(user_id)
        h = Holding(
            user_id=uid, asset=asset,
            position_size=position_size,
            avg_cost=avg_cost,
            horizon=horizon,
            notes=notes,
            updated_at=time.time(),
        )
        self._data.setdefault(uid, {})[asset] = h
        self._save()
        return h

    def clear(self, user_id: str | int, asset: str) -> bool:
        """Remove a holding entry. Returns True if it existed."""
        self._ensure_loaded()
        uid = str(user_id)
        user_holdings = self._data.get(uid, {})
        if asset in user_holdings:
            del user_holdings[asset]
            self._save()
            return True
        return False

    def reload(self) -> None:
        """Test hook."""
        self._data.clear()
        self._loaded = False


# ─── Holdings context block ───────────────────────────────────────────────────

def build_holdings_context_block(
    user_id: str | int | None,
    asset: str | None,
    store: HoldingsStore | None = None,
) -> str:
    """
    Return a prompt-injectable text block describing user's holdings for `asset`.
    Returns "" if no asset / no holding found.
    """
    if user_id is None or asset is None:
        return ""
    s = store or default_holdings_store()
    holding = s.get(user_id, asset)
    if holding is None:
        return f"[Holding] {asset}: no position recorded (use /setholding to tell me if you're already in this)"
    return holding.to_context_block()


def holdings_reply_addendum(
    user_id: str | int | None,
    asset: str | None,
    store: HoldingsStore | None = None,
    current_price: float | None = None,
) -> str:
    """
    Returns a short addendum sentence to append to decision reply,
    reflecting the user's holding context including P&L status.

    - no position  → 偏"是否适合建仓"措辞
    - in_profit    → 提示是否适合持有或分批止盈
    - underwater   → 提示是否继续持有或减仓观察
    - near_cost    → 提示接近成本价，关注方向确认
    - heavy + any  → 加重风险提示
    """
    if user_id is None or asset is None:
        return ""
    s = store or default_holdings_store()
    holding = s.get(user_id, asset)
    if holding is None:
        return ""
    if not holding.has_position:
        return "(You don't have a position here. If you're thinking of starting one, begin small.)"

    pnl = holding.pnl_status(current_price)

    if holding.is_heavy:
        base = "(Heads-up: you're already sized heavy. Adding more is risky — prioritize exposure management. "
        if pnl == "in_profit":
            return base + "Currently in profit — consider scaling out in tranches.)"
        if pnl == "underwater":
            return base + "Currently underwater — tightly control downside risk.)"
        return base + ")"

    # small / medium
    if pnl == "in_profit":
        cost_note = f" (cost {holding.avg_cost:,.2f})" if holding.avg_cost else ""
        return (f"(You hold a {holding.position_size} position in {asset}{cost_note}, "
                f"currently in profit — use the analysis above to decide whether to hold or scale out in tranches.)")
    if pnl == "underwater":
        cost_note = f" (cost {holding.avg_cost:,.2f})" if holding.avg_cost else ""
        return (f"(You hold a {holding.position_size} position in {asset}{cost_note}, "
                f"currently underwater — factor the analysis above into the hold-or-trim decision.)")
    if pnl == "near_cost":
        return (f"(You hold a {holding.position_size} position in {asset}, "
                f"currently near your cost basis — key inflection, wait for direction to confirm before acting.)")
    # unknown pnl (no cost price)
    return f"(You hold a {holding.position_size} position in {asset}; use the analysis above to decide whether to add.)"


# ─── Singleton ────────────────────────────────────────────────────────────────

_default_holdings: HoldingsStore | None = None


def default_holdings_store() -> HoldingsStore:
    global _default_holdings
    if _default_holdings is None:
        _default_holdings = HoldingsStore()
    return _default_holdings


def reset_holdings_store() -> None:
    """Test hook."""
    global _default_holdings
    _default_holdings = None


# ─── Telegram command helpers ─────────────────────────────────────────────────

def parse_setholding_args(
    text: str,
) -> tuple[str | None, PositionSize, float | None, Horizon]:
    """
    Parse "/setholding <asset> [size] [cost] [horizon]"

    Examples:
      /setholding gold                → ("gold", "small", None, "unknown")
      /setholding gold medium         → ("gold", "medium", None, "unknown")
      /setholding gold medium 3320    → ("gold", "medium", 3320.0, "unknown")
      /setholding gold medium 3320 long → ("gold", "medium", 3320.0, "long")

    Returns (asset, size, avg_cost, horizon).
    """
    parts = text.strip().split()
    # parts[0] = "/setholding"
    asset = parts[1].lower() if len(parts) > 1 else None

    valid_sizes: list[PositionSize] = ["none", "small", "medium", "large"]
    valid_horizons: list[Horizon] = ["short", "mid", "long", "unknown"]

    size: PositionSize = "small"
    avg_cost: float | None = None
    horizon: Horizon = "unknown"

    for token in parts[2:]:
        lower = token.lower()
        if lower in valid_sizes:
            size = lower  # type: ignore[assignment]
        elif lower in valid_horizons:
            horizon = lower  # type: ignore[assignment]
        else:
            try:
                avg_cost = float(token.replace(",", ""))
            except ValueError:
                pass

    return asset, size, avg_cost, horizon


def parse_clearholding_args(text: str) -> str | None:
    """
    Parse "/clearholding gold" → "gold"
    """
    parts = text.strip().split()
    return parts[1].lower() if len(parts) > 1 else None
