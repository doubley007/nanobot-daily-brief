"""
资产别名表 —— query_router / RAG / decision_engine 共用。

Key 设计原则：
  - 规范化 id（小写下划线），供下游统一引用
  - 中英文别名都放进来，规则层能命中
  - keywords 用于 RAG 粗检索（命中任一即视为相关）
  - tickers 为后续接入价格数据预留
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssetSpec:
    id: str
    display_name: str
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    category: str = "other"  # equity / commodity / crypto / fx / index


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        id="gold",
        display_name="黄金",
        aliases=("黄金", "金价", "xau", "xauusd", "gold", "金子"),
        keywords=("gold", "xau", "bullion", "黄金", "金价", "safe haven", "avaro"),
        tickers=("GC=F", "GLD"),
        category="commodity",
    ),
    AssetSpec(
        id="bitcoin",
        display_name="比特币",
        aliases=("比特币", "btc", "bitcoin", "xbt"),
        keywords=("bitcoin", "btc", "比特币", "crypto"),
        tickers=("BTC-USD",),
        category="crypto",
    ),
    AssetSpec(
        id="tesla",
        display_name="特斯拉",
        aliases=("特斯拉", "tesla", "tsla"),
        keywords=("tesla", "tsla", "musk", "特斯拉"),
        tickers=("TSLA",),
        category="equity",
    ),
    AssetSpec(
        id="nvidia",
        display_name="英伟达",
        aliases=("英伟达", "nvidia", "nvda", "老黄"),
        keywords=("nvidia", "nvda", "ai chip", "gpu", "英伟达"),
        tickers=("NVDA",),
        category="equity",
    ),
    AssetSpec(
        id="sp500",
        display_name="标普 500",
        aliases=("标普", "标普500", "sp500", "s&p", "spx", "美股", "大盘"),
        keywords=("s&p 500", "s&p500", "spx", "标普", "美股", "sp500"),
        tickers=("^GSPC", "SPY"),
        category="index",
    ),
    AssetSpec(
        id="usd",
        display_name="美元",
        aliases=("美元", "usd", "dxy", "dollar"),
        keywords=("usd", "dollar", "美元", "dxy"),
        tickers=("DX-Y.NYB",),
        category="fx",
    ),
    AssetSpec(
        id="oil",
        display_name="原油",
        aliases=("原油", "石油", "crude", "oil", "wti", "brent"),
        keywords=("oil", "crude", "wti", "brent", "原油", "opec"),
        tickers=("CL=F",),
        category="commodity",
    ),
    AssetSpec(
        id="ethereum",
        display_name="以太坊",
        aliases=("以太坊", "eth", "ethereum", "以太"),
        keywords=("ethereum", "eth", "以太坊", "以太", "defi", "eip"),
        tickers=("ETH-USD",),
        category="crypto",
    ),
    AssetSpec(
        id="silver",
        display_name="白银",
        aliases=("白银", "银价", "silver", "xag", "xagusd"),
        keywords=("silver", "xag", "白银", "银价"),
        tickers=("SI=F", "SLV"),
        category="commodity",
    ),
    AssetSpec(
        id="a_shares",
        display_name="A股",
        aliases=("A股", "a股", "沪深", "沪深300", "csi300", "上证", "深证", "创业板", "科创板"),
        keywords=("a股", "沪深", "沪深300", "上证", "深证", "创业板", "科创板", "a-share", "csi"),
        tickers=("000300.SS", "^SSEC"),
        category="index",
    ),
    AssetSpec(
        id="hk_stocks",
        display_name="港股",
        aliases=("港股", "恒生", "恒指", "hsi", "hang seng", "hk", "香港股市"),
        keywords=("港股", "恒生", "恒指", "hsi", "hang seng", "香港"),
        tickers=("^HSI",),
        category="index",
    ),
    AssetSpec(
        id="sti",
        display_name="新加坡海峡时报指数",
        aliases=("sti", "海峡时报指数", "新加坡大盘", "新加坡指数", "新交所", "sgx", "strait times index"),
        keywords=("sti", "strait times index", "sgx", "新加坡", "singapore index", "新加坡大盘"),
        tickers=("^STI",),
        category="index",
    ),
    AssetSpec(
        id="dbs",
        display_name="星展银行",
        aliases=("dbs", "星展", "星展银行", "d05"),
        keywords=("dbs", "星展", "d05.si", "singapore bank", "星展银行"),
        tickers=("D05.SI",),
        category="equity",
    ),
    AssetSpec(
        id="ocbc",
        display_name="华侨银行",
        aliases=("ocbc", "华侨", "华侨银行", "o39"),
        keywords=("ocbc", "华侨", "o39.si", "oversea-chinese banking"),
        tickers=("O39.SI",),
        category="equity",
    ),
    AssetSpec(
        id="uob",
        display_name="大华银行",
        aliases=("uob", "大华", "大华银行", "u11"),
        keywords=("uob", "大华", "u11.si", "united overseas bank"),
        tickers=("U11.SI",),
        category="equity",
    ),
    AssetSpec(
        id="cict",
        display_name="凯德综合商业信托",
        aliases=("cict", "凯德", "c38u", "capitaland integrated commercial trust"),
        keywords=("cict", "凯德", "c38u.si", "capitaland", "singapore reit", "新加坡reit"),
        tickers=("C38U.SI",),
        category="equity",
    ),
    AssetSpec(
        id="mapletree_pan_asia",
        display_name="丰树泛亚商业信托",
        aliases=("mapletree", "mct", "n2iu", "丰树", "mapletree pan asia"),
        keywords=("mapletree", "mct", "n2iu.si", "丰树", "reit singapore"),
        tickers=("N2IU.SI",),
        category="equity",
    ),
    AssetSpec(
        id="sgd",
        display_name="新加坡元",
        aliases=("sgd", "新币", "新加坡元", "新加坡币", "s$"),
        keywords=("sgd", "新币", "新加坡元", "singapore dollar", "usdsgd"),
        tickers=("SGD=X",),
        category="fx",
    ),
    AssetSpec(
        id="copper",
        display_name="铜",
        aliases=("铜", "copper", "铜价", "hg"),
        keywords=("copper", "铜", "铜价", "hg futures", "industrial metal"),
        tickers=("HG=F",),
        category="commodity",
    ),
    AssetSpec(
        id="nasdaq",
        display_name="纳斯达克",
        aliases=("纳斯达克", "nasdaq", "ndx", "qqq", "科技股", "纳指"),
        keywords=("nasdaq", "ndx", "qqq", "纳斯达克", "纳指", "tech index"),
        tickers=("^IXIC", "QQQ"),
        category="index",
    ),
)


_ASSETS_BY_ID: dict[str, AssetSpec] = {a.id: a for a in ASSETS}


def known_asset_ids() -> list[str]:
    return [a.id for a in ASSETS]


def get_asset(asset_id: str) -> AssetSpec | None:
    return _ASSETS_BY_ID.get(asset_id)


def detect_asset(text: str) -> str | None:
    """
    纯规则匹配 —— 找出第一个命中别名/关键词的资产。
    如果一条消息命中多个资产，返回别名最长的那个（"黄金 vs 金" 优先黄金）。
    """
    if not text:
        return None
    lower = text.lower()
    best_id: str | None = None
    best_len = 0
    for spec in ASSETS:
        all_terms = set(spec.aliases) | set(spec.keywords)
        for term in all_terms:
            t = term.lower()
            if t and t in lower and len(t) > best_len:
                best_id = spec.id
                best_len = len(t)
    return best_id


def asset_display(asset_id: str | None) -> str:
    if not asset_id:
        return "未识别资产"
    spec = get_asset(asset_id)
    return spec.display_name if spec else asset_id


def asset_keywords(asset_id: str) -> list[str]:
    spec = get_asset(asset_id)
    return list(spec.keywords) if spec else []
