"""
Demo CLI —— 支持黄金和比特币场景。

用法：
    PYTHONPATH=app .venv/bin/python -m assistant.demo "我能不能买黄金？"
    PYTHONPATH=app .venv/bin/python -m assistant.demo --no-llm "黄金还能追吗？"
    PYTHONPATH=app .venv/bin/python -m assistant.demo --asset bitcoin "比特币能买吗？"
    PYTHONPATH=app .venv/bin/python -m assistant.demo --trace "大家都在买黄金"
    PYTHONPATH=app .venv/bin/python -m assistant.demo --debug "我能不能买黄金？"

默认会写入/覆盖 assistant fixture，这样即使知识库还没 ingest 真数据，
也能看到完整链路的回答。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from assistant.fixtures import install_gold_fixture, install_bitcoin_fixture
import assistant.pipeline as pipeline
from assistant.pipeline import answer_question_traced
from assistant.trend_signals import trend_from_values
from assistant.telegram_bot import _format_why_block

_ASSET_STUBS = {
    "gold": (0.03, 0.09),
    "bitcoin": (0.05, 0.18),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+",
                        help="你想问 bot 的问题（会被拼接成一句话）")
    parser.add_argument("--no-llm", action="store_true",
                        help="强制不走 LLM（只看规则/骨架输出）")
    parser.add_argument("--no-fixture", action="store_true",
                        help="不预装 fixture（用当前 store 真实数据）")
    parser.add_argument("--asset", choices=["gold", "bitcoin"], default="gold",
                        help="使用哪个资产的 fixture（默认 gold）")
    parser.add_argument("--trace", action="store_true",
                        help="除了 reply 还打印整条 trace 的 JSON")
    parser.add_argument("--debug", action="store_true",
                        help="启用 ASSISTANT_DEBUG，在日志里打印完整路由/决策轨迹")
    parser.add_argument("--explain", action="store_true",
                        help="打印决策推理骨架（等同于 Telegram /why 的输出）")
    parser.add_argument("--report", action="store_true",
                        help="生成并打印资产快照报告（等同于 Telegram /report <asset>）")
    parser.add_argument("--report-style", choices=["analyst", "executive"], default="analyst",
                        help="报告风格：analyst（详细）或 executive（简洁摘要）")
    args = parser.parse_args(argv)

    if args.debug:
        os.environ["ASSISTANT_DEBUG"] = "1"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    if not args.no_fixture:
        if args.asset == "bitcoin":
            n_news, n_com = install_bitcoin_fixture()
        else:
            n_news, n_com = install_gold_fixture()
        print(f"[fixture:{args.asset}] installed news={n_news}, community={n_com}")

    # 绕过 yfinance 网络（demo 重点是整条链路）
    r7, r30 = _ASSET_STUBS.get(args.asset, (0.03, 0.09))
    pipeline.fetch_trend_signal = lambda asset: trend_from_values(asset, r7=r7, r30=r30)
    if args.no_llm:
        pipeline._llm_callable = lambda: None

    question = " ".join(args.question)
    trace = answer_question_traced(question)

    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(trace.reply)
    print()

    if args.explain:
        print("--- EXPLAIN (Decision Skeleton) ---")
        print(_format_why_block(trace))
        print()

    if args.report:
        style_label = args.report_style.upper()
        print(f"--- REPORT (Market Snapshot / {style_label}) ---")
        from assistant.report import generate_report
        print(generate_report(args.asset, style=args.report_style))
        print()

    if args.trace:
        print("--- TRACE ---")
        payload = {
            "route": trace.route.to_dict(),
            "emotion": trace.emotion.to_dict(),
            "decision": trace.decision.to_dict() if trace.decision else None,
            "aggregate": trace.aggregate.to_dict() if trace.aggregate else None,
            "trend": trace.trend.to_dict() if trace.trend else None,
            # v2 additions
            "company": trace.company.to_dict() if trace.company else None,
            "profile": trace.profile.to_dict() if trace.profile else None,
            "policy_violations": trace.policy_violations,
            "context_debug": trace.context_pkg.to_debug_dict() if trace.context_pkg else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
