"""
Evaluation runner for the nanobot market assistant pipeline.

Usage:
    cd /Users/yangyang/nanobot_project
    python -m app.evaluation.run_eval [--questions data/eval_questions.json]
                                      [--out-json reports/eval_report.json]
                                      [--out-md   reports/eval_report.md]
                                      [--timeout  45]
                                      [--db       /tmp/eval_kb.sqlite3]

The runner calls assistant.pipeline.answer_question_traced for every question,
extracts evidence-quality fields from the PipelineTrace, runs rule-based scoring
via evaluation.scorer, writes JSON + Markdown reports, and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# ─── path setup: run_eval.py lives in app/evaluation/, project root is ../.. ──
_PROJ = Path(__file__).resolve().parent.parent.parent   # nanobot_project/
_APP  = _PROJ / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from evaluation.scorer import score_result  # noqa: E402


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation")

_DEFAULT_QUESTIONS = _PROJ / "data" / "eval_questions.json"
_DEFAULT_JSON_OUT  = _PROJ / "reports" / "eval_report.json"
_DEFAULT_MD_OUT    = _PROJ / "reports" / "eval_report.md"
_DEFAULT_TIMEOUT   = 60   # seconds per question


# ─── pipeline wrapper ─────────────────────────────────────────────────────────

def _run_one(question_entry: dict[str, Any], db_path: str) -> dict[str, Any]:
    """
    Run one question through the pipeline and return a raw result dict.
    Isolated to its own function so ThreadPoolExecutor can call it safely.
    """
    # Point the knowledge DB at the eval-specific file (avoids polluting prod data)
    os.environ["ASSISTANT_KNOWLEDGE_DB"] = db_path
    os.environ["SESSION_MEMORY_FILE"]    = db_path.replace(".sqlite3", "_sessions.json")

    # Reset singletons so env var takes effect in each thread
    try:
        from assistant.rag import store as _store_mod
        _store_mod._default = None
        from assistant.session_memory import reset_session_store
        reset_session_store()
    except Exception:
        pass

    question = question_entry["question"]
    category = question_entry.get("category", "unknown")

    t0 = time.time()
    try:
        from assistant.pipeline import answer_question_traced
        trace = answer_question_traced(question, user_id=None)
        elapsed = time.time() - t0
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "id": question_entry.get("id", "?"),
            "question": question,
            "category": category,
            "expected_route": question_entry.get("expected_route"),
            "expected_asset": question_entry.get("expected_asset"),
            "answer": f"[ERROR] {type(exc).__name__}: {exc}",
            "route": "error",
            "asset": None,
            "direct_answer_present": False,
            "has_trend_data": False,
            "citations_count": 0,
            "elapsed_s": round(elapsed, 2),
            "error": traceback.format_exc(),
        }

    # Extract evidence signals from trace
    pkg = trace.context_pkg
    news_count  = len(pkg.news)      if pkg else 0
    comm_count  = len(pkg.community) if pkg else 0
    has_trend   = trace.trend is not None
    citations   = []
    if pkg and pkg.news:
        seen_urls: set[str] = set()
        for doc in pkg.news[:5]:
            url = getattr(doc, "url", None) or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            title = getattr(doc, "title", "") or ""
            source = getattr(doc, "source", "") or ""
            if title:
                citations.append({"title": title, "source": source, "url": url})

    # direct_answer: check if the MCP server would have produced one
    from evaluation.scorer import _FACT_KEYWORDS  # reuse the same keyword list
    q_lower = question.lower()
    is_fact_q = any(kw in question or kw in q_lower for kw in _FACT_KEYWORDS)
    has_direct = (
        has_trend
        and is_fact_q
        and trace.trend is not None
        and (
            getattr(trace.trend, "recent_return_7d", None) is not None
            or getattr(trace.trend, "recent_return_30d", None) is not None
        )
    )

    return {
        "id": question_entry.get("id", "?"),
        "question": question,
        "category": category,
        "expected_route": question_entry.get("expected_route"),
        "expected_asset": question_entry.get("expected_asset"),
        "answer": trace.reply,
        "route": trace.route.route,
        "asset": trace.route.asset,
        "direct_answer_present": has_direct,
        "has_trend_data": has_trend,
        "citations_count": len(citations),
        "citations": citations,
        "news_count": news_count,
        "community_count": comm_count,
        "decision_action": trace.decision.action if trace.decision else None,
        "decision_confidence": trace.decision.confidence if trace.decision else None,
        "build_ms": round(trace.meta.get("context_build_ms", 0), 1) if trace.meta else 0,
        "elapsed_s": round(elapsed, 2),
    }


# ─── report generators ────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{n / total * 100:.0f}%"


def build_markdown_report(
    scored: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> str:
    total = len(scored)
    if total == 0:
        return "# Eval Report\n\nNo results.\n"

    avg_score = sum(r["total_score"] for r in scored) / total
    max_score = 10

    # Aggregate flags
    n_direct         = sum(1 for r in scored if r.get("direct_answer_used"))
    n_factual        = sum(1 for r in scored if r.get("category") == "factual")
    n_risk           = sum(1 for r in scored if r.get("has_risk_warning"))
    n_confidence     = sum(1 for r in scored if r.get("has_confidence_level"))
    n_route_correct  = sum(1 for r in scored if r.get("route_correct"))
    n_asset_correct  = sum(1 for r in scored if r.get("asset_correct"))
    n_hallucination  = sum(1 for r in scored if r.get("hallucination_risk"))
    n_has_mdata      = sum(1 for r in scored if r.get("has_market_data"))
    n_has_action     = sum(1 for r in scored if r.get("has_action_suggestion"))

    # Per-category breakdown
    categories = sorted(set(r.get("category", "unknown") for r in scored))
    cat_stats: dict[str, dict[str, Any]] = {}
    for cat in categories:
        items = [r for r in scored if r.get("category") == cat]
        if not items:
            continue
        cat_stats[cat] = {
            "count": len(items),
            "avg_score": sum(r["total_score"] for r in items) / len(items),
            "min_score": min(r["total_score"] for r in items),
            "max_score": max(r["total_score"] for r in items),
        }

    # High-hallucination items
    hall_items = [r for r in scored if r.get("hallucination_risk")]

    # Low-score items (for actionable insight)
    low_items = sorted(scored, key=lambda r: r["total_score"])[:5]

    now_str = run_meta.get("run_at", datetime.now().isoformat()[:19])
    lines = [
        f"# NanoBot Pipeline Evaluation Report",
        f"",
        f"**Run at:** {now_str}  ",
        f"**Questions:** {total}  ",
        f"**Max score per question:** {max_score}  ",
        f"",
        f"## Overall Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average score | **{avg_score:.2f} / {max_score}** |",
        f"| Route accuracy | {n_route_correct}/{total} ({_pct(n_route_correct, total)}) |",
        f"| Asset accuracy | {n_asset_correct}/{total} ({_pct(n_asset_correct, total)}) |",
        f"| direct_answer coverage (factual) | {n_direct}/{n_factual} ({_pct(n_direct, n_factual)}) |",
        f"| Market data present | {n_has_mdata}/{total} ({_pct(n_has_mdata, total)}) |",
        f"| Risk warning present | {n_risk}/{total} ({_pct(n_risk, total)}) |",
        f"| Action suggestion present | {n_has_action}/{total} ({_pct(n_has_action, total)}) |",
        f"| Confidence level present | {n_confidence}/{total} ({_pct(n_confidence, total)}) |",
        f"| **Hallucination risk flagged** | **{n_hallucination}/{total} ({_pct(n_hallucination, total)})** |",
        f"",
        f"## Score by Category",
        f"",
        f"| Category | Count | Avg Score | Min | Max |",
        f"|----------|-------|-----------|-----|-----|",
    ]
    for cat, s in cat_stats.items():
        lines.append(
            f"| {cat} | {s['count']} | {s['avg_score']:.2f} | {s['min_score']} | {s['max_score']} |"
        )

    lines += [
        f"",
        f"## Hallucination Risk Items ({n_hallucination})",
        f"",
    ]
    if hall_items:
        for r in hall_items:
            lines.append(f"- **[{r['id']}]** {r['question']}")
            lines.append(f"  - Route: `{r['route']}` | Asset: `{r.get('asset')}`")
            ans_snip = (r.get("answer") or "")[:120].replace("\n", " ")
            lines.append(f"  - Answer snippet: _{ans_snip}_")
    else:
        lines.append("_None flagged._")

    lines += [
        f"",
        f"## 5 Lowest-Scoring Questions",
        f"",
        f"| ID | Question | Score | Route | Asset | Issue |",
        f"|----|----------|-------|-------|-------|-------|",
    ]
    for r in low_items:
        issues = []
        if not r.get("has_clear_conclusion"):
            issues.append("no_conclusion")
        if not r.get("has_risk_warning"):
            issues.append("no_risk")
        if not r.get("has_action_suggestion"):
            issues.append("no_action")
        if not r.get("route_correct"):
            issues.append(f"route_wrong({r.get('route')}≠{r.get('expected_route')})")
        if r.get("hallucination_risk"):
            issues.append("hallucination")
        issue_str = " ".join(issues) or "—"
        q_short = r["question"][:40] + ("…" if len(r["question"]) > 40 else "")
        lines.append(
            f"| {r['id']} | {q_short} | {r['total_score']} | `{r['route']}` | `{r.get('asset')}` | {issue_str} |"
        )

    lines += [
        f"",
        f"## All Results",
        f"",
        f"| ID | Cat | Score | Conclusion | Risk | Action | Conf | DA | Route✓ | Asset✓ | Hall | Q |",
        f"|----|-----|-------|------------|------|--------|------|----|----|------|------|---|",
    ]

    def _yesno(v: bool | None) -> str:
        return "✓" if v else "✗"

    for r in sorted(scored, key=lambda x: (x["category"], x["id"])):
        q_short = r["question"][:35] + ("…" if len(r["question"]) > 35 else "")
        lines.append(
            f"| {r['id']} | {r['category'][:4]} | {r['total_score']} "
            f"| {_yesno(r.get('has_clear_conclusion'))} "
            f"| {_yesno(r.get('has_risk_warning'))} "
            f"| {_yesno(r.get('has_action_suggestion'))} "
            f"| {_yesno(r.get('has_confidence_level'))} "
            f"| {_yesno(r.get('direct_answer_used'))} "
            f"| {_yesno(r.get('route_correct'))} "
            f"| {_yesno(r.get('asset_correct'))} "
            f"| {_yesno(r.get('hallucination_risk'))} "
            f"| {q_short} |"
        )

    lines += ["", "---", "_Generated by `app/evaluation/run_eval.py`_", ""]
    return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pipeline evaluation")
    parser.add_argument("--questions", default=str(_DEFAULT_QUESTIONS))
    parser.add_argument("--out-json", default=str(_DEFAULT_JSON_OUT))
    parser.add_argument("--out-md",   default=str(_DEFAULT_MD_OUT))
    parser.add_argument("--timeout",  type=int, default=_DEFAULT_TIMEOUT)
    parser.add_argument("--workers",  type=int, default=1,
                        help="Parallel workers (default 1; >1 may hit SQLite concurrency)")
    parser.add_argument("--db",       default="",
                        help="Path for eval SQLite DB (default: temp file)")
    args = parser.parse_args(argv)

    # Load questions
    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"ERROR: questions file not found: {q_path}", file=sys.stderr)
        return 1
    with open(q_path, encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)
    print(f"Loaded {len(questions)} questions from {q_path}")

    # Ensure output dirs exist
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    # DB path for eval (isolated from prod)
    if args.db:
        db_path = args.db
    else:
        db_fd, db_path = tempfile.mkstemp(suffix="_eval.sqlite3", prefix="nanobot_eval_")
        os.close(db_fd)

    print(f"Using eval DB: {db_path}")
    print(f"Running {len(questions)} questions (workers={args.workers}, timeout={args.timeout}s)...")
    print()

    run_at = datetime.now().isoformat()[:19]
    raw_results: list[dict[str, Any]] = []

    if args.workers == 1:
        for i, q in enumerate(questions, 1):
            print(f"  [{i:02d}/{len(questions)}] {q['question'][:60]}", end="", flush=True)
            result = _run_one(q, db_path)
            raw_results.append(result)
            status = "ERROR" if "error" in result else f"route={result['route']} asset={result['asset']}"
            print(f"  →  {status}  ({result['elapsed_s']}s)")
    else:
        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for q in questions:
                fut = pool.submit(_run_one, q, db_path)
                futures_map[fut] = q
            for fut in as_completed(futures_map):
                q = futures_map[fut]
                try:
                    raw_results.append(fut.result(timeout=args.timeout))
                except Exception as exc:
                    raw_results.append({
                        "id": q.get("id", "?"),
                        "question": q["question"],
                        "category": q.get("category", "unknown"),
                        "expected_route": q.get("expected_route"),
                        "expected_asset": q.get("expected_asset"),
                        "answer": f"[TIMEOUT/ERROR] {exc}",
                        "route": "error",
                        "asset": None,
                        "direct_answer_present": False,
                        "has_trend_data": False,
                        "citations_count": 0,
                        "elapsed_s": args.timeout,
                    })

    # Score
    print()
    print("Scoring results...")
    scored = [score_result(r) for r in raw_results]

    # Sort back to original question order
    id_order = {q["id"]: i for i, q in enumerate(questions)}
    scored.sort(key=lambda r: id_order.get(r.get("id", ""), 999))

    # Build report metadata
    run_meta = {
        "run_at": run_at,
        "total_questions": len(scored),
        "avg_score": round(sum(r["total_score"] for r in scored) / max(len(scored), 1), 2),
        "eval_db": db_path,
    }

    # Write JSON
    out_json = {"meta": run_meta, "results": scored}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON report written: {args.out_json}")

    # Write Markdown
    md_text = build_markdown_report(scored, run_meta)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Markdown report written: {args.out_md}")

    # Console summary
    total = len(scored)
    avg   = run_meta["avg_score"]
    n_hall = sum(1 for r in scored if r.get("hallucination_risk"))
    n_route_ok = sum(1 for r in scored if r.get("route_correct"))
    n_direct = sum(1 for r in scored if r.get("direct_answer_used"))
    n_factual = sum(1 for r in scored if r.get("category") == "factual")
    n_risk = sum(1 for r in scored if r.get("has_risk_warning"))

    print()
    print("=" * 50)
    print(f"  Questions     : {total}")
    print(f"  Avg score     : {avg:.2f} / 10")
    print(f"  Route acc.    : {n_route_ok}/{total} ({n_route_ok/total*100:.0f}%)")
    print(f"  direct_answer : {n_direct}/{n_factual} factual questions")
    print(f"  Risk warning  : {n_risk}/{total} ({n_risk/total*100:.0f}%)")
    print(f"  Hallucination : {n_hall}/{total} flagged")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
