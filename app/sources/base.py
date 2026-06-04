"""
所有社区源 adapter 都实现同一个接口 —— 返回一组 UnifiedPost。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from community.schema import UnifiedPost


@dataclass
class FetchReport:
    platform: str
    ok: bool
    posts: list[UnifiedPost]
    error: str = ""

    @property
    def count(self) -> int:
        return len(self.posts)


class BaseSourceAdapter(ABC):
    """所有社区数据源适配器共同基类。"""

    platform: str = "unknown"

    @abstractmethod
    def is_configured(self) -> bool:
        """检查当前环境变量是否配齐。未配齐的源不会进入抓取循环。"""

    @abstractmethod
    def fetch(self) -> FetchReport:
        """抓取并返回标准化好的 UnifiedPost 列表。永远不抛异常。"""

    # 默认实现：调用子类 fetch() 并吞所有异常，保证 orchestration 稳定
    def safe_fetch(self) -> FetchReport:
        try:
            return self.fetch()
        except Exception as e:  # noqa: BLE001
            return FetchReport(
                platform=self.platform, ok=False, posts=[], error=f"{type(e).__name__}: {e}",
            )
