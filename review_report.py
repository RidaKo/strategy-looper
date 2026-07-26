from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import strategy_research as sr


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


def load_latest_report() -> tuple[dict[str, object], Path]:
    candidates = []
    latest = RESULTS_DIR / "latest_strategy_search.json"
    if latest.exists():
        candidates.append(latest)
    candidates.extend(
        sorted(
            RESULTS_DIR.glob("strategy_search_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    )
    seen = set()
    errors = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate.read_text(encoding="utf-8")), candidate
        except json.JSONDecodeError as exc:
            errors.append(f"{candidate.name}: {exc}")
    raise RuntimeError("No valid JSON report found. " + "; ".join(errors[:3]))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def num(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def chart_value(value: object) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    return float(value)


def html_table(
    rows: list[dict[str, object]],
    columns: list[str] | None = None,
    limit: int = 20,
) -> str:
    rows = rows[:limit]
    if not rows:
        return "<p>No rows.</p>"
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in columns)
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def points_for_series(values: list[float], width: int, height: int, pad: int):
    if not values:
        return "", 0.0, 0.0
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    points = []
    for idx, value in enumerate(values):
        x = pad + idx * (width - 2 * pad) / max(1, len(values) - 1)
        y = height - pad - (value - lo) * (height - 2 * pad) / (hi - lo)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points), lo, hi


def line_chart(
    series: list[tuple[object, float]],
    title: str,
    width: int = 980,
    height: int = 320,
    color: str = "#1f77b4",
) -> str:
    values = [float(value) for _, value in series]
    points, lo, hi = points_for_series(values, width, height, 36)
    if not points:
        return "<p>No chart data.</p>"
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      <text x="20" y="22" font-size="16" font-family="Arial">{escape(title)}</text>
      <line x1="36" y1="{height-36}" x2="{width-20}" y2="{height-36}" stroke="#bbb"/>
      <line x1="36" y1="36" x2="36" y2="{height-36}" stroke="#bbb"/>
      <text x="42" y="50" font-size="11" font-family="Arial">{hi:.2f}</text>
      <text x="42" y="{height-44}" font-size="11" font-family="Arial">{lo:.2f}</text>
      <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>
    </svg>"""


def bar_chart(
    rows: list[dict[str, object]],
    label_key: str,
    value_key: str,
    title: str,
    width: int = 980,
    height: int = 360,
    color: str = "#4c78a8",
) -> str:
    if not rows:
        return "<p>No chart data.</p>"
    labels = [str(row[label_key]) for row in rows]
    values = [chart_value(row[value_key]) for row in rows]
    lo = min(0.0, min(values))
    hi = max(0.0, max(values))
    if math.isclose(lo, hi):
        hi = lo + 1.0
    pad_l, pad_b, pad_t = 56, 80, 36
    plot_w = width - pad_l - 20
    plot_h = height - pad_t - pad_b
    bar_w = plot_w / max(1, len(values)) * 0.72
    zero_y = pad_t + (hi - 0.0) * plot_h / (hi - lo)
    bars = []
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = pad_l + idx * plot_w / max(1, len(values)) + bar_w * 0.15
        y = pad_t + (hi - max(value, 0.0)) * plot_h / (hi - lo)
        h = abs(value) * plot_h / (hi - lo)
        if value < 0:
            y = zero_y
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>'
        )
        label_x = x + bar_w / 2
        bars.append(
            f'<text x="{label_x:.1f}" y="{height-18}" font-size="10" text-anchor="middle" '
            f'transform="rotate(45 {label_x:.1f},{height-18})" font-family="Arial">{escape(label[:18])}</text>'
        )
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      <text x="20" y="22" font-size="16" font-family="Arial">{escape(title)}</text>
      <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width-20}" y2="{zero_y:.1f}" stroke="#999"/>
      <text x="8" y="{pad_t+12}" font-size="11" font-family="Arial">{hi:.2f}</text>
      <text x="8" y="{height-pad_b}" font-size="11" font-family="Arial">{lo:.2f}</text>
      {''.join(bars)}
    </svg>"""


def variant_from_dict(data: dict[str, object]) -> sr.StrategyVariant:
    return sr.StrategyVariant(
        family=str(data["family"]),
        params=tuple(sorted(data["params"].items())),
        side=str(data["side"]),
    )


def aggregate_daily_series(
    runs: dict[str, sr.MarketRun], symbols: list[str]
) -> list[tuple[object, float]]:
    bucket: dict[object, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for symbol in symbols:
        run = runs[symbol]
        for one_date, value in zip(run.dates, run.returns, strict=True):
            bucket[one_date][0] += value
            bucket[one_date][1] += 1.0
    return [
        (one_date, total / count if count else 0.0)
        for one_date, (total, count) in sorted(bucket.items())
    ]


def equity_and_drawdown(
    return_series: list[tuple[object, float]]
) -> tuple[list[tuple[object, float]], list[tuple[object, float]]]:
    equity = 1.0
    high = 1.0
    equity_series = []
    drawdown_series = []
    for one_date, value in return_series:
        equity *= max(0.0, 1.0 + value)
        high = max(high, equity)
        drawdown = (high - equity) / high if high else 0.0
        equity_series.append((one_date, equity))
        drawdown_series.append((one_date, drawdown))
    return equity_series, drawdown_series


def build_report_html() -> str:
    report, report_path = load_latest_report()
    found = report.get("found")
    robustness = report.get("robustness") or {}
    walk_forward = report.get("walk_forward") or {}
    top_rows = read_csv_rows(RESULTS_DIR / "latest_strategy_search.csv")
    failed_rows = read_csv_rows(RESULTS_DIR / "latest_strategy_search_failed.csv")
    robust_rows = read_csv_rows(RESULTS_DIR / "latest_strategy_search_robustness.csv")
    walk_rows = read_csv_rows(RESULTS_DIR / "latest_strategy_search_walk_forward.csv")

    sections = []
    sections.append(f"<p><b>Source report:</b> {escape(str(report_path))}</p>")

    if found:
        summary = [
            {
                "strategy": found["strategy_label"],
                "universe": found["universe"],
                "test_sharpe": num(found["test"]["sharpe"]),
                "test_ann_return": pct(found["test"]["annual_return"]),
                "test_max_dd": pct(found["test"]["max_drawdown"]),
                "positive_test_markets": pct(found["positive_test_market_fraction"]),
                "robustness": robustness.get("status", "not_run"),
                "walk_forward": walk_forward.get("status", "not_run"),
            }
        ]
        sections.append("<h2>Selected Candidate</h2>" + html_table(summary))
        sections.append(
            "<p><b>Robustness warnings:</b> "
            + escape(", ".join(robustness.get("warnings", [])) or "none")
            + "</p>"
        )
        sections.append(
            "<p><b>Walk-forward warnings:</b> "
            + escape(", ".join(walk_forward.get("warnings", [])) or "none")
            + "</p>"
        )

    sections.append(
        "<h2>Top Strategies</h2>"
        + html_table(
            top_rows,
            [
                "rank",
                "passed",
                "universe",
                "strategy",
                "test_sharpe",
                "test_annual_return",
                "test_max_drawdown",
                "positive_test_market_fraction",
                "failed_reasons",
            ],
            limit=15,
        )
    )

    reason_counts = Counter()
    for row in failed_rows:
        for reason in row.get("failed_reasons", "").split(";"):
            if reason:
                reason_counts[reason] += 1
    reason_rows = [
        {"reason": key, "count": value} for key, value in reason_counts.most_common(12)
    ]
    sections.append("<h2>Failure Reasons</h2>" + html_table(reason_rows))
    sections.append(bar_chart(reason_rows, "reason", "count", "Failed strategy reason counts"))

    if found:
        markets = sr.load_markets(DATA_DIR, min_bars=750)
        universes = sr.build_universes(markets)
        market_by_symbol = {market.symbol: market for market in markets}
        variant = variant_from_dict(found["strategy"])
        symbols = universes[found["universe"]]
        selected_markets = [market_by_symbol[symbol] for symbol in symbols]
        runs = sr.backtest_variant(selected_markets, variant, report["criteria"]["cost_bps"])
        daily_returns = aggregate_daily_series(runs, symbols)
        equity_series, drawdown_series = equity_and_drawdown(daily_returns)
        sections.append("<h2>Equity Curve</h2>" + line_chart(equity_series, "Selected strategy equity curve"))
        sections.append(
            "<h2>Drawdown</h2>"
            + line_chart(drawdown_series, "Selected strategy drawdown", color="#d62728")
        )

        split_rows = []
        for split in ["train", "validation", "test"]:
            metrics = found[split]
            split_rows.append(
                {
                    "split": split,
                    "days": metrics["days"],
                    "total_return": pct(metrics["total_return"]),
                    "annual_return": pct(metrics["annual_return"]),
                    "sharpe": num(metrics["sharpe"]),
                    "max_drawdown": pct(metrics["max_drawdown"]),
                    "win_rate": pct(metrics["win_rate"]),
                    "profit_factor": num(metrics["profit_factor"]),
                }
            )
        sections.append("<h2>Train / Validation / Test</h2>" + html_table(split_rows))
        sections.append(bar_chart(split_rows, "split", "annual_return", "Annual return by split"))
        sections.append(
            bar_chart(split_rows, "split", "max_drawdown", "Max drawdown by split", color="#d62728")
        )

    costs = [row for row in robust_rows if row.get("check_type") == "cost_stress"]
    leave_one = [row for row in robust_rows if row.get("check_type") == "leave_one_out"]
    single_market = [row for row in robust_rows if row.get("check_type") == "single_market"]
    sections.append(
        "<h2>Cost Stress</h2>"
        + html_table(
            costs,
            ["item", "passed", "test_sharpe", "test_annual_return", "test_max_drawdown", "failed_reasons"],
        )
    )
    sections.append(
        "<h2>Leave One Market Out</h2>"
        + html_table(
            leave_one,
            ["item", "passed", "test_sharpe", "test_annual_return", "test_max_drawdown", "failed_reasons"],
        )
    )
    sections.append(
        "<h2>Single Market Breakdown</h2>"
        + html_table(
            single_market,
            ["universe", "passed", "test_sharpe", "test_annual_return", "test_max_drawdown", "failed_reasons"],
        )
    )

    selected_folds = [
        row for row in walk_rows if row.get("scope") == "selected_universe_optimized"
    ]
    sections.append(
        "<h2>Walk-Forward Folds</h2>"
        + html_table(
            selected_folds,
            [
                "fold",
                "test_start",
                "test_end",
                "optimized_strategy",
                "optimized_test_sharpe",
                "optimized_test_annual_return",
                "confirmed_test_sharpe",
                "confirmed_test_annual_return",
                "confirmed_test_max_drawdown",
            ],
            limit=30,
        )
    )
    sections.append(
        bar_chart(
            selected_folds,
            "fold",
            "confirmed_test_annual_return",
            "Confirmed strategy walk-forward annual returns",
        )
    )
    sections.append(
        bar_chart(
            selected_folds,
            "fold",
            "confirmed_test_sharpe",
            "Confirmed strategy walk-forward Sharpe",
        )
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>StrategyLooper Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0 24px; }}
    td, th {{ border: 1px solid #ddd; padding: 4px 6px; text-align: right; }}
    th {{ text-align: center; background: #f5f5f5; }}
    h1, h2 {{ margin-top: 28px; }}
    svg {{ max-width: 100%; height: auto; border: 1px solid #eee; margin: 8px 0 24px; }}
  </style>
</head>
<body>
<h1>StrategyLooper Review</h1>
{''.join(sections)}
</body>
</html>"""
    return html


def write_html_report(path: Path | None = None) -> Path:
    if path is None:
        path = RESULTS_DIR / "latest_strategy_review.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report_html(), encoding="utf-8")
    return path


def main() -> int:
    path = write_html_report()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
