from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class MarketData:
    symbol: str
    description: str
    exchange: str
    tick_size: float
    big_point_value: float
    dates: list[date]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]

    @property
    def bars(self) -> int:
        return len(self.closes)

    @property
    def root(self) -> str:
        return normalize_root(self.symbol)


@dataclass(frozen=True)
class StrategyVariant:
    family: str
    params: tuple[tuple[str, float | int], ...]
    side: str

    def get(self, name: str) -> float | int:
        return dict(self.params)[name]

    def label(self) -> str:
        params = ", ".join(f"{key}={value}" for key, value in self.params)
        return f"{self.family}({params}, side={self.side})"

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "params": {key: value for key, value in self.params},
            "side": self.side,
        }


@dataclass
class MarketRun:
    symbol: str
    dates: list[date]
    returns: list[float]
    trades: int
    exposure_days: int


@dataclass
class Metrics:
    days: int
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    daily_volatility: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "days": self.days,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "daily_volatility": self.daily_volatility,
        }


@dataclass
class Evaluation:
    variant: StrategyVariant
    universe: str
    market_count: int
    train: Metrics
    validation: Metrics
    test: Metrics
    total_trades: int
    exposure_fraction: float
    positive_test_market_fraction: float
    score: float
    passed: bool
    failed_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.variant.to_dict(),
            "strategy_label": self.variant.label(),
            "universe": self.universe,
            "market_count": self.market_count,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "total_trades": self.total_trades,
            "exposure_fraction": self.exposure_fraction,
            "positive_test_market_fraction": self.positive_test_market_fraction,
            "score": self.score,
            "passed": self.passed,
            "failed_reasons": list(self.failed_reasons),
        }


@dataclass(frozen=True)
class SearchConfig:
    data_dir: Path
    results_dir: Path
    cost_bps: float
    max_trials: int
    time_budget_sec: float
    seed: int
    stop_on_pass: bool
    min_bars: int
    min_markets: int
    min_oos_days: int
    min_total_trades: int
    min_exposure_fraction: float
    target_test_sharpe: float
    target_test_annual_return: float
    max_test_drawdown: float
    min_validation_sharpe: float
    min_train_sharpe: float
    min_positive_market_fraction: float
    top_count: int
    run_post_confirmation_checks: bool
    robustness_cost_multipliers: tuple[float, ...]
    walk_forward_train_days: int
    walk_forward_test_days: int
    walk_forward_step_days: int
    walk_forward_max_variants: int


def excel_datetime_to_date(value: float) -> date:
    timestamp = round((value - 25569.0) * 86400.0)
    return datetime.fromtimestamp(timestamp, UTC).date()


def read_string(handle) -> str:
    raw_size = handle.read(4)
    if len(raw_size) != 4:
        raise ValueError("Unexpected end of file while reading string length")
    size = struct.unpack("<i", raw_size)[0]
    return handle.read(size).decode("ascii")


def read_market_file(path: Path) -> MarketData:
    with path.open("rb") as handle:
        magic, format_type = struct.unpack("<ii", handle.read(8))
        if magic != 1111111111 or format_type != 3:
            raise ValueError(f"{path} is not a supported OHLCV .dat file")

        tick_size = struct.unpack("<d", handle.read(8))[0]
        big_point_value = struct.unpack("<d", handle.read(8))[0]
        _country = read_string(handle)
        exchange = read_string(handle)
        symbol = read_string(handle)
        description = read_string(handle)
        _interval_type = read_string(handle)
        _interval_span = struct.unpack("<i", handle.read(4))[0]
        _time_zone = read_string(handle)
        _session = read_string(handle)
        payload = handle.read()

    record_size = struct.calcsize("<dddddd")
    usable_size = len(payload) - (len(payload) % record_size)
    dates: list[date] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    for raw_date, open_, high, low, close, volume in struct.iter_unpack(
        "<dddddd", payload[:usable_size]
    ):
        if open_ <= 0 or high <= 0 or low <= 0 or close <= 0:
            continue
        if high < max(open_, close) or low > min(open_, close):
            continue
        dates.append(excel_datetime_to_date(raw_date))
        opens.append(open_)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        volumes.append(volume)

    if len(closes) < 2:
        raise ValueError(f"{path} has too few usable bars")

    return MarketData(
        symbol=symbol,
        description=description,
        exchange=exchange,
        tick_size=tick_size,
        big_point_value=big_point_value,
        dates=dates,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
    )


def load_markets(data_dir: Path, min_bars: int) -> list[MarketData]:
    markets: list[MarketData] = []
    for path in sorted(data_dir.glob("*.dat")):
        market = read_market_file(path)
        if market.bars >= min_bars:
            markets.append(market)
    if not markets:
        raise RuntimeError(f"No usable .dat files found in {data_dir}")
    return markets


def normalize_root(symbol: str) -> str:
    symbol = symbol.strip().lstrip("@")
    return symbol.split("=")[0]


def rolling_mean(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    total = 0.0
    for idx, value in enumerate(values):
        total += value
        if idx >= period:
            total -= values[idx - period]
        if idx >= period - 1:
            result[idx] = total / period
    return result


def rolling_mean_stdev(
    values: list[float], period: int
) -> tuple[list[float | None], list[float | None]]:
    means: list[float | None] = [None] * len(values)
    stdevs: list[float | None] = [None] * len(values)
    total = 0.0
    total_sq = 0.0
    for idx, value in enumerate(values):
        total += value
        total_sq += value * value
        if idx >= period:
            old = values[idx - period]
            total -= old
            total_sq -= old * old
        if idx >= period - 1:
            mean = total / period
            variance = max(total_sq / period - mean * mean, 0.0)
            means[idx] = mean
            stdevs[idx] = math.sqrt(variance)
    return means, stdevs


def rolling_extreme(values: list[float], period: int, find_max: bool) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    indexes: deque[int] = deque()

    for idx, value in enumerate(values):
        while indexes and indexes[0] <= idx - period:
            indexes.popleft()
        if find_max:
            while indexes and values[indexes[-1]] <= value:
                indexes.pop()
        else:
            while indexes and values[indexes[-1]] >= value:
                indexes.pop()
        indexes.append(idx)
        if idx >= period - 1:
            result[idx] = values[indexes[0]]

    return result


def apply_side(position: int, side: str) -> int:
    if side == "both":
        return position
    if side == "inverse":
        return -position
    if side == "long":
        return max(position, 0)
    if side == "short":
        return min(position, 0)
    raise ValueError(f"Unknown side: {side}")


def momentum_positions(market: MarketData, variant: StrategyVariant) -> list[int]:
    lookback = int(variant.get("lookback"))
    threshold = float(variant.get("threshold"))
    closes = market.closes
    positions = [0] * len(closes)

    for idx in range(lookback + 1, len(closes)):
        signal_idx = idx - 1
        base = closes[signal_idx - lookback]
        if base <= 0:
            continue
        change = closes[signal_idx] / base - 1.0
        raw_position = 1 if change > threshold else -1 if change < -threshold else 0
        positions[idx] = apply_side(raw_position, variant.side)
    return positions


def moving_average_positions(market: MarketData, variant: StrategyVariant) -> list[int]:
    fast = int(variant.get("fast"))
    slow = int(variant.get("slow"))
    band = float(variant.get("band"))
    closes = market.closes
    fast_ma = rolling_mean(closes, fast)
    slow_ma = rolling_mean(closes, slow)
    positions = [0] * len(closes)

    for idx in range(slow + 1, len(closes)):
        signal_idx = idx - 1
        fast_value = fast_ma[signal_idx]
        slow_value = slow_ma[signal_idx]
        if fast_value is None or slow_value is None or slow_value == 0:
            continue
        spread = fast_value / slow_value - 1.0
        raw_position = 1 if spread > band else -1 if spread < -band else 0
        positions[idx] = apply_side(raw_position, variant.side)
    return positions


def donchian_positions(market: MarketData, variant: StrategyVariant) -> list[int]:
    lookback = int(variant.get("lookback"))
    closes = market.closes
    high_channel = rolling_extreme(market.highs, lookback, find_max=True)
    low_channel = rolling_extreme(market.lows, lookback, find_max=False)
    positions = [0] * len(closes)
    held = 0

    for idx in range(lookback + 1, len(closes)):
        signal_idx = idx - 1
        prior_high = high_channel[signal_idx - 1]
        prior_low = low_channel[signal_idx - 1]
        if prior_high is None or prior_low is None:
            positions[idx] = apply_side(held, variant.side)
            continue
        if closes[signal_idx] > prior_high:
            held = 1
        elif closes[signal_idx] < prior_low:
            held = -1
        positions[idx] = apply_side(held, variant.side)
    return positions


def zscore_reversion_positions(market: MarketData, variant: StrategyVariant) -> list[int]:
    lookback = int(variant.get("lookback"))
    entry = float(variant.get("entry"))
    exit_z = float(variant.get("exit"))
    closes = market.closes
    means, stdevs = rolling_mean_stdev(closes, lookback)
    positions = [0] * len(closes)
    held = 0

    for idx in range(lookback + 1, len(closes)):
        signal_idx = idx - 1
        mean = means[signal_idx]
        stdev = stdevs[signal_idx]
        if mean is None or stdev is None or stdev == 0:
            continue
        z_score = (closes[signal_idx] - mean) / stdev
        if held == 0:
            if z_score > entry:
                held = -1
            elif z_score < -entry:
                held = 1
        elif held == 1 and z_score >= -exit_z:
            held = 0
        elif held == -1 and z_score <= exit_z:
            held = 0
        positions[idx] = apply_side(held, variant.side)
    return positions


def volume_confirmed_momentum_positions(
    market: MarketData, variant: StrategyVariant
) -> list[int]:
    lookback = int(variant.get("lookback"))
    volume_lookback = int(variant.get("volume_lookback"))
    threshold = float(variant.get("threshold"))
    volume_multiple = float(variant.get("volume_multiple"))
    closes = market.closes
    volume_ma = rolling_mean(market.volumes, volume_lookback)
    positions = [0] * len(closes)

    start = max(lookback, volume_lookback) + 1
    for idx in range(start, len(closes)):
        signal_idx = idx - 1
        base = closes[signal_idx - lookback]
        avg_volume = volume_ma[signal_idx]
        if base <= 0 or avg_volume is None or avg_volume <= 0:
            continue
        change = closes[signal_idx] / base - 1.0
        volume_ok = market.volumes[signal_idx] >= avg_volume * volume_multiple
        if not volume_ok:
            continue
        raw_position = 1 if change > threshold else -1 if change < -threshold else 0
        positions[idx] = apply_side(raw_position, variant.side)
    return positions


def build_positions(market: MarketData, variant: StrategyVariant) -> list[int]:
    if variant.family == "momentum":
        return momentum_positions(market, variant)
    if variant.family == "moving_average":
        return moving_average_positions(market, variant)
    if variant.family == "donchian":
        return donchian_positions(market, variant)
    if variant.family == "zscore_reversion":
        return zscore_reversion_positions(market, variant)
    if variant.family == "volume_momentum":
        return volume_confirmed_momentum_positions(market, variant)
    raise ValueError(f"Unknown strategy family: {variant.family}")


def backtest_market(
    market: MarketData, variant: StrategyVariant, cost_bps: float
) -> MarketRun:
    positions = build_positions(market, variant)
    returns: list[float] = []
    dates: list[date] = []
    trades = 0
    exposure_days = 0
    previous_position = 0
    cost_rate = cost_bps / 10000.0

    for idx in range(1, market.bars):
        current_position = positions[idx]
        previous_close = market.closes[idx - 1]
        current_close = market.closes[idx]
        raw_return = 0.0
        if previous_close > 0:
            raw_return = current_close / previous_close - 1.0
        strategy_return = current_position * raw_return
        position_change = abs(current_position - previous_position)
        if position_change:
            trades += 1
            strategy_return -= position_change * cost_rate
        if current_position:
            exposure_days += 1
        returns.append(strategy_return)
        dates.append(market.dates[idx])
        previous_position = current_position

    return MarketRun(
        symbol=market.symbol,
        dates=dates,
        returns=returns,
        trades=trades,
        exposure_days=exposure_days,
    )


def metrics_from_returns(returns: list[float]) -> Metrics:
    days = len(returns)
    if days == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    equity = 1.0
    high_water = 1.0
    max_drawdown = 0.0
    wins = 0
    positive_sum = 0.0
    negative_sum = 0.0

    for value in returns:
        if value > 0:
            wins += 1
            positive_sum += value
        elif value < 0:
            negative_sum += -value
        equity *= max(0.0, 1.0 + value)
        high_water = max(high_water, equity)
        if high_water > 0:
            max_drawdown = max(max_drawdown, (high_water - equity) / high_water)

    total_return = equity - 1.0
    if equity > 0:
        annual_return = equity ** (TRADING_DAYS_PER_YEAR / days) - 1.0
    else:
        annual_return = -1.0

    mean = sum(returns) / days
    variance = sum((value - mean) ** 2 for value in returns) / days
    daily_volatility = math.sqrt(variance)
    sharpe = 0.0
    if daily_volatility > 0:
        sharpe = math.sqrt(TRADING_DAYS_PER_YEAR) * mean / daily_volatility
    win_rate = wins / days
    profit_factor = positive_sum / negative_sum if negative_sum > 0 else math.inf

    return Metrics(
        days=days,
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        daily_volatility=daily_volatility * math.sqrt(TRADING_DAYS_PER_YEAR),
    )


def choose_split_dates(markets: list[MarketData]) -> tuple[date, date]:
    all_dates = sorted({one_date for market in markets for one_date in market.dates})
    if len(all_dates) < 100:
        raise RuntimeError("Not enough dates for train/validation/test split")
    train_end = all_dates[int(len(all_dates) * 0.60)]
    validation_end = all_dates[int(len(all_dates) * 0.80)]
    return train_end, validation_end


def split_name(one_date: date, train_end: date, validation_end: date) -> str:
    if one_date <= train_end:
        return "train"
    if one_date <= validation_end:
        return "validation"
    return "test"


def aggregate_returns(
    runs: dict[str, MarketRun],
    symbols: list[str],
    train_end: date,
    validation_end: date,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    buckets: dict[str, dict[date, list[float]]] = {
        "train": defaultdict(lambda: [0.0, 0.0]),
        "validation": defaultdict(lambda: [0.0, 0.0]),
        "test": defaultdict(lambda: [0.0, 0.0]),
    }
    active_counts = {"train": 0, "validation": 0, "test": 0}

    for symbol in symbols:
        run = runs[symbol]
        seen_split = set()
        for one_date, value in zip(run.dates, run.returns, strict=True):
            name = split_name(one_date, train_end, validation_end)
            bucket = buckets[name][one_date]
            bucket[0] += value
            bucket[1] += 1.0
            seen_split.add(name)
        for name in seen_split:
            active_counts[name] += 1

    split_returns: dict[str, list[float]] = {}
    for name, bucket in buckets.items():
        split_returns[name] = [
            total / count for _, (total, count) in sorted(bucket.items()) if count > 0
        ]
    return split_returns, active_counts


def positive_market_fraction(
    runs: dict[str, MarketRun],
    symbols: list[str],
    train_end: date,
    validation_end: date,
) -> float:
    positives = 0
    active = 0
    for symbol in symbols:
        run = runs[symbol]
        test_returns = [
            value
            for one_date, value in zip(run.dates, run.returns, strict=True)
            if split_name(one_date, train_end, validation_end) == "test"
        ]
        if len(test_returns) < 100:
            continue
        active += 1
        if metrics_from_returns(test_returns).total_return > 0:
            positives += 1
    return positives / active if active else 0.0


def aggregate_returns_between(
    runs: dict[str, MarketRun],
    symbols: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[list[float], int]:
    bucket: dict[date, list[float]] = defaultdict(lambda: [0.0, 0.0])
    active_symbols = 0

    for symbol in symbols:
        run = runs[symbol]
        saw_data = False
        for one_date, value in zip(run.dates, run.returns, strict=True):
            if start_date is not None and one_date < start_date:
                continue
            if end_date is not None and one_date > end_date:
                continue
            bucket[one_date][0] += value
            bucket[one_date][1] += 1.0
            saw_data = True
        if saw_data:
            active_symbols += 1

    returns = [
        total / count for _, (total, count) in sorted(bucket.items()) if count > 0
    ]
    return returns, active_symbols


def metrics_between(
    runs: dict[str, MarketRun],
    symbols: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[Metrics, int]:
    returns, active_symbols = aggregate_returns_between(
        runs, symbols, start_date, end_date
    )
    return metrics_from_returns(returns), active_symbols


def fraction(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(1 for value in values_list if value) / len(values_list)


def evaluate_universe(
    variant: StrategyVariant,
    universe: str,
    symbols: list[str],
    runs: dict[str, MarketRun],
    train_end: date,
    validation_end: date,
    config: SearchConfig,
    min_markets: int | None = None,
) -> Evaluation | None:
    required_min_markets = config.min_markets if min_markets is None else min_markets
    if len(symbols) < required_min_markets:
        return None

    split_returns, active_counts = aggregate_returns(
        runs, symbols, train_end, validation_end
    )
    if active_counts["test"] < required_min_markets:
        return None

    train = metrics_from_returns(split_returns["train"])
    validation = metrics_from_returns(split_returns["validation"])
    test = metrics_from_returns(split_returns["test"])

    total_trades = sum(runs[symbol].trades for symbol in symbols)
    exposure_days = sum(runs[symbol].exposure_days for symbol in symbols)
    total_days = sum(len(runs[symbol].returns) for symbol in symbols)
    exposure_fraction = exposure_days / total_days if total_days else 0.0
    positive_fraction = positive_market_fraction(
        runs, symbols, train_end, validation_end
    )

    failed_reasons = rejection_reasons(
        train=train,
        validation=validation,
        test=test,
        total_trades=total_trades,
        exposure_fraction=exposure_fraction,
        positive_fraction=positive_fraction,
        config=config,
    )
    passed = not failed_reasons

    score = (
        test.sharpe * 2.0
        + validation.sharpe
        + train.sharpe * 0.35
        + test.annual_return * 6.0
        + positive_fraction
        - test.max_drawdown * 2.5
        + min(total_trades / 2000.0, 0.5)
    )

    return Evaluation(
        variant=variant,
        universe=universe,
        market_count=len(symbols),
        train=train,
        validation=validation,
        test=test,
        total_trades=total_trades,
        exposure_fraction=exposure_fraction,
        positive_test_market_fraction=positive_fraction,
        score=score,
        passed=passed,
        failed_reasons=tuple(failed_reasons),
    )


def rejection_reasons(
    train: Metrics,
    validation: Metrics,
    test: Metrics,
    total_trades: int,
    exposure_fraction: float,
    positive_fraction: float,
    config: SearchConfig,
) -> list[str]:
    reasons: list[str] = []
    if test.days < config.min_oos_days:
        reasons.append(f"test_days<{config.min_oos_days}")
    if total_trades < config.min_total_trades:
        reasons.append(f"trades<{config.min_total_trades}")
    if exposure_fraction < config.min_exposure_fraction:
        reasons.append(f"exposure<{config.min_exposure_fraction:g}")
    if train.sharpe < config.min_train_sharpe:
        reasons.append(f"train_sharpe<{config.min_train_sharpe:g}")
    if validation.sharpe < config.min_validation_sharpe:
        reasons.append(f"validation_sharpe<{config.min_validation_sharpe:g}")
    if validation.annual_return <= 0:
        reasons.append("validation_annual_return<=0")
    if test.sharpe < config.target_test_sharpe:
        reasons.append(f"test_sharpe<{config.target_test_sharpe:g}")
    if test.annual_return < config.target_test_annual_return:
        reasons.append(f"test_annual_return<{config.target_test_annual_return:g}")
    if test.max_drawdown > config.max_test_drawdown:
        reasons.append(f"test_drawdown>{config.max_test_drawdown:g}")
    if positive_fraction < config.min_positive_market_fraction:
        reasons.append(f"positive_market_fraction<{config.min_positive_market_fraction:g}")
    return reasons


def build_universes(markets: list[MarketData]) -> dict[str, list[str]]:
    by_symbol = {market.symbol: market for market in markets}

    groups = {
        "all": set(by_symbol),
        "all_ex_vx_btc": {s for s, m in by_symbol.items() if m.root not in {"VX", "BTC"}},
        "equity_index": {"ES", "NQ", "YM", "RTY", "EMD", "NK", "FESX", "FDAX"},
        "rates": {
            "TU",
            "FV",
            "TY",
            "US",
            "UB",
            "FGBL",
            "FGBM",
            "FGBS",
            "FGBX",
            "TEN",
            "ULS",
            "LT2",
            "LZ",
        },
        "currencies": {
            "AD",
            "BP",
            "CD",
            "EC",
            "JY",
            "SF",
            "DX",
            "MP1",
            "NE1",
            "BR",
            "DA",
        },
        "energies": {"CL", "HO", "RB", "NG", "BRN", "WBS"},
        "metals": {"GC", "SI", "HG", "PL", "PA"},
        "grains_oilseeds": {"C", "W", "S", "SM", "BO", "KW", "O", "RR"},
        "softs": {"CC", "CC3", "KC", "SB", "CT", "OJ"},
        "livestock": {"LC", "LH", "FC"},
        "commodities": {
            "CL",
            "HO",
            "RB",
            "NG",
            "BRN",
            "WBS",
            "GC",
            "SI",
            "HG",
            "PL",
            "PA",
            "C",
            "W",
            "S",
            "SM",
            "BO",
            "KW",
            "O",
            "RR",
            "CC",
            "CC3",
            "KC",
            "SB",
            "CT",
            "OJ",
            "LC",
            "LH",
            "FC",
        },
    }

    universes: dict[str, list[str]] = {}
    for name, roots_or_symbols in groups.items():
        if name.startswith("all"):
            selected = sorted(roots_or_symbols)
        else:
            selected = sorted(
                symbol
                for symbol, market in by_symbol.items()
                if market.root in roots_or_symbols
            )
        if selected:
            universes[name] = selected
    return universes


def make_variant(
    family: str, side: str, **params: float | int
) -> StrategyVariant:
    ordered = tuple(sorted(params.items()))
    return StrategyVariant(family=family, params=ordered, side=side)


def grid_variants() -> Iterable[StrategyVariant]:
    sides = ("both", "inverse", "long", "short")

    for lookback in (20, 40, 60, 90, 120, 180, 252):
        for threshold in (0.0, 0.02, 0.05, 0.10):
            for side in sides:
                yield make_variant(
                    "momentum", side, lookback=lookback, threshold=threshold
                )

    for fast in (5, 10, 20, 40, 60):
        for slow in (50, 100, 150, 200, 300):
            if fast >= slow:
                continue
            for band in (0.0, 0.005, 0.01):
                for side in sides:
                    yield make_variant(
                        "moving_average", side, fast=fast, slow=slow, band=band
                    )

    for lookback in (20, 40, 60, 90, 120, 180, 252):
        for side in sides:
            yield make_variant("donchian", side, lookback=lookback)

    for lookback in (20, 40, 60, 100, 150):
        for entry in (1.0, 1.5, 2.0, 2.5):
            for exit_z in (0.0, 0.25, 0.5):
                for side in sides:
                    yield make_variant(
                        "zscore_reversion",
                        side,
                        lookback=lookback,
                        entry=entry,
                        exit=exit_z,
                    )

    for lookback in (20, 40, 60, 90, 120):
        for volume_lookback in (20, 60, 120):
            for threshold in (0.0, 0.02, 0.05):
                for volume_multiple in (0.8, 1.0, 1.2):
                    for side in sides:
                        yield make_variant(
                            "volume_momentum",
                            side,
                            lookback=lookback,
                            volume_lookback=volume_lookback,
                            threshold=threshold,
                            volume_multiple=volume_multiple,
                        )


def random_variant(rng: random.Random) -> StrategyVariant:
    side = rng.choice(("both", "inverse", "long", "short"))
    family = rng.choice(
        (
            "momentum",
            "moving_average",
            "donchian",
            "zscore_reversion",
            "volume_momentum",
        )
    )
    if family == "momentum":
        return make_variant(
            family,
            side,
            lookback=rng.randint(10, 320),
            threshold=round(rng.uniform(0.0, 0.15), 4),
        )
    if family == "moving_average":
        fast = rng.randint(3, 80)
        slow = rng.randint(max(fast + 5, 30), 360)
        return make_variant(
            family,
            side,
            fast=fast,
            slow=slow,
            band=round(rng.uniform(0.0, 0.025), 4),
        )
    if family == "donchian":
        return make_variant(family, side, lookback=rng.randint(10, 320))
    if family == "zscore_reversion":
        return make_variant(
            family,
            side,
            lookback=rng.randint(10, 220),
            entry=round(rng.uniform(0.8, 3.0), 2),
            exit=round(rng.uniform(0.0, 0.8), 2),
        )
    return make_variant(
        family,
        side,
        lookback=rng.randint(10, 240),
        volume_lookback=rng.randint(10, 180),
        threshold=round(rng.uniform(0.0, 0.12), 4),
        volume_multiple=round(rng.uniform(0.6, 1.5), 2),
    )


def variant_stream(seed: int) -> Iterable[StrategyVariant]:
    seen: set[StrategyVariant] = set()
    for variant in grid_variants():
        seen.add(variant)
        yield variant

    rng = random.Random(seed)
    while True:
        variant = random_variant(rng)
        if variant in seen:
            continue
        seen.add(variant)
        yield variant


def backtest_variant(
    markets: list[MarketData], variant: StrategyVariant, cost_bps: float
) -> dict[str, MarketRun]:
    return {
        market.symbol: backtest_market(market, variant, cost_bps)
        for market in markets
    }


def unique_variants(variants: Iterable[StrategyVariant]) -> list[StrategyVariant]:
    seen: set[StrategyVariant] = set()
    result: list[StrategyVariant] = []
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        result.append(variant)
    return result


def bounded_int(value: float, minimum: int = 2, maximum: int = 500) -> int:
    return max(minimum, min(maximum, int(round(value))))


def variant_neighbors(variant: StrategyVariant) -> list[StrategyVariant]:
    params = dict(variant.params)
    side = variant.side
    candidates: list[StrategyVariant] = []

    if variant.family == "momentum":
        lookback = int(params["lookback"])
        threshold = float(params["threshold"])
        lookbacks = [bounded_int(lookback * mult) for mult in (0.75, 0.9, 1.0, 1.1, 1.25)]
        thresholds = sorted(
            {
                max(0.0, round(threshold + offset, 4))
                for offset in (-0.03, -0.01, 0.0, 0.01, 0.03)
            }
        )
        for one_lookback in lookbacks:
            for one_threshold in thresholds:
                candidates.append(
                    make_variant(
                        "momentum",
                        side,
                        lookback=one_lookback,
                        threshold=one_threshold,
                    )
                )

    elif variant.family == "moving_average":
        fast = int(params["fast"])
        slow = int(params["slow"])
        band = float(params["band"])
        fasts = [bounded_int(fast * mult) for mult in (0.75, 1.0, 1.25)]
        slows = [bounded_int(slow * mult) for mult in (0.8, 1.0, 1.2)]
        bands = sorted(
            {
                max(0.0, round(band + offset, 4))
                for offset in (-0.005, 0.0, 0.005)
            }
        )
        for one_fast in fasts:
            for one_slow in slows:
                if one_fast >= one_slow:
                    continue
                for one_band in bands:
                    candidates.append(
                        make_variant(
                            "moving_average",
                            side,
                            fast=one_fast,
                            slow=one_slow,
                            band=one_band,
                        )
                    )

    elif variant.family == "donchian":
        lookback = int(params["lookback"])
        for one_lookback in [bounded_int(lookback * mult) for mult in (0.75, 0.9, 1.0, 1.1, 1.25)]:
            candidates.append(make_variant("donchian", side, lookback=one_lookback))

    elif variant.family == "zscore_reversion":
        lookback = int(params["lookback"])
        entry = float(params["entry"])
        exit_z = float(params["exit"])
        lookbacks = [bounded_int(lookback * mult) for mult in (0.75, 1.0, 1.25)]
        entries = [max(0.1, round(entry + offset, 2)) for offset in (-0.3, 0.0, 0.3)]
        exits = [max(0.0, round(exit_z + offset, 2)) for offset in (-0.2, 0.0, 0.2)]
        for one_lookback in lookbacks:
            for one_entry in entries:
                for one_exit in exits:
                    candidates.append(
                        make_variant(
                            "zscore_reversion",
                            side,
                            lookback=one_lookback,
                            entry=one_entry,
                            exit=one_exit,
                        )
                    )

    elif variant.family == "volume_momentum":
        lookback = int(params["lookback"])
        volume_lookback = int(params["volume_lookback"])
        threshold = float(params["threshold"])
        volume_multiple = float(params["volume_multiple"])
        lookbacks = [bounded_int(lookback * mult) for mult in (0.75, 1.0, 1.25)]
        volume_lookbacks = [
            bounded_int(volume_lookback * mult) for mult in (0.75, 1.0, 1.25)
        ]
        thresholds = [max(0.0, round(threshold + offset, 4)) for offset in (-0.02, 0.0, 0.02)]
        volume_multiples = [
            max(0.1, round(volume_multiple + offset, 2))
            for offset in (-0.2, 0.0, 0.2)
        ]
        for one_lookback in lookbacks:
            for one_volume_lookback in volume_lookbacks:
                for one_threshold in thresholds:
                    for one_volume_multiple in volume_multiples:
                        candidates.append(
                            make_variant(
                                "volume_momentum",
                                side,
                                lookback=one_lookback,
                                volume_lookback=one_volume_lookback,
                                threshold=one_threshold,
                                volume_multiple=one_volume_multiple,
                            )
                        )

    return unique_variants(candidates)


def summarize_evaluations(evaluations: list[Evaluation]) -> dict[str, object]:
    if not evaluations:
        return {
            "count": 0,
            "passed_count": 0,
            "pass_fraction": 0.0,
            "profitable_fraction": 0.0,
        }

    test_sharpes = [item.test.sharpe for item in evaluations]
    test_returns = [item.test.annual_return for item in evaluations]
    test_drawdowns = [item.test.max_drawdown for item in evaluations]
    return {
        "count": len(evaluations),
        "passed_count": sum(1 for item in evaluations if item.passed),
        "pass_fraction": fraction(item.passed for item in evaluations),
        "profitable_fraction": fraction(item.test.annual_return > 0 for item in evaluations),
        "median_test_sharpe": median(test_sharpes),
        "mean_test_sharpe": mean(test_sharpes),
        "median_test_annual_return": median(test_returns),
        "mean_test_annual_return": mean(test_returns),
        "worst_test_drawdown": max(test_drawdowns),
    }


def run_robustness_tests(
    markets: list[MarketData],
    selected: Evaluation,
    config: SearchConfig,
) -> dict[str, object]:
    universes = build_universes(markets)
    symbols = universes[selected.universe]
    market_by_symbol = {market.symbol: market for market in markets}
    selected_markets = [market_by_symbol[symbol] for symbol in symbols]
    train_end, validation_end = choose_split_dates(markets)
    minimum_markets = max(1, min(config.min_markets, len(symbols)))

    cost_tests: list[dict[str, object]] = []
    for multiplier in config.robustness_cost_multipliers:
        stressed_config = replace(config, cost_bps=config.cost_bps * multiplier)
        runs = backtest_variant(selected_markets, selected.variant, stressed_config.cost_bps)
        evaluation = evaluate_universe(
            selected.variant,
            selected.universe,
            symbols,
            runs,
            train_end,
            validation_end,
            stressed_config,
            min_markets=minimum_markets,
        )
        if evaluation is not None:
            cost_tests.append(
                {
                    "cost_multiplier": multiplier,
                    "cost_bps": stressed_config.cost_bps,
                    "evaluation": evaluation.to_dict(),
                }
            )

    neighbor_evaluations: list[Evaluation] = []
    for neighbor in variant_neighbors(selected.variant):
        runs = backtest_variant(selected_markets, neighbor, config.cost_bps)
        evaluation = evaluate_universe(
            neighbor,
            selected.universe,
            symbols,
            runs,
            train_end,
            validation_end,
            config,
            min_markets=minimum_markets,
        )
        if evaluation is not None:
            neighbor_evaluations.append(evaluation)
    neighbor_evaluations.sort(key=lambda item: item.score, reverse=True)

    selected_runs = backtest_variant(selected_markets, selected.variant, config.cost_bps)
    leave_one_out: list[dict[str, object]] = []
    for omitted_symbol in symbols:
        subset = [symbol for symbol in symbols if symbol != omitted_symbol]
        if not subset:
            continue
        evaluation = evaluate_universe(
            selected.variant,
            f"{selected.universe}_minus_{omitted_symbol}",
            subset,
            selected_runs,
            train_end,
            validation_end,
            config,
            min_markets=max(1, min(minimum_markets, len(subset))),
        )
        if evaluation is not None:
            leave_one_out.append(
                {
                    "omitted_symbol": omitted_symbol,
                    "evaluation": evaluation.to_dict(),
                }
            )

    market_breakdown: list[dict[str, object]] = []
    for symbol in symbols:
        evaluation = evaluate_universe(
            selected.variant,
            symbol,
            [symbol],
            selected_runs,
            train_end,
            validation_end,
            config,
            min_markets=1,
        )
        if evaluation is not None:
            market_breakdown.append(evaluation.to_dict())

    warnings: list[str] = []
    if len(symbols) <= 4:
        warnings.append("selected_universe_has_four_or_fewer_markets")
    if selected.train.max_drawdown > config.max_test_drawdown:
        warnings.append("train_drawdown_exceeds_test_drawdown_limit")
    if selected.validation.max_drawdown > config.max_test_drawdown:
        warnings.append("validation_drawdown_exceeds_test_drawdown_limit")

    neighbor_summary = summarize_evaluations(neighbor_evaluations)
    if neighbor_summary["pass_fraction"] < 0.50:
        warnings.append("less_than_half_of_parameter_neighbors_passed")
    if fraction(item["evaluation"]["test"]["annual_return"] > 0 for item in leave_one_out) < 0.75:
        warnings.append("leave_one_out_profitability_below_75_percent")

    cost_pass_fraction = fraction(item["evaluation"]["passed"] for item in cost_tests)
    robustness_passed = (
        cost_pass_fraction >= 0.50
        and neighbor_summary["pass_fraction"] >= 0.50
        and not any(
            warning in warnings
            for warning in (
                "leave_one_out_profitability_below_75_percent",
                "selected_universe_has_four_or_fewer_markets",
            )
        )
    )

    return {
        "status": "passed" if robustness_passed else "flagged",
        "warnings": warnings,
        "cost_tests": cost_tests,
        "cost_pass_fraction": cost_pass_fraction,
        "parameter_neighbor_summary": neighbor_summary,
        "parameter_neighbors": [item.to_dict() for item in neighbor_evaluations],
        "top_parameter_neighbors": [
            item.to_dict() for item in neighbor_evaluations[:10]
        ],
        "bottom_parameter_neighbors": [
            item.to_dict() for item in neighbor_evaluations[-10:]
        ],
        "leave_one_out": leave_one_out,
        "market_breakdown": market_breakdown,
    }


def period_score(metrics: Metrics) -> float:
    if metrics.days == 0:
        return -math.inf
    return (
        metrics.sharpe * 2.0
        + metrics.annual_return * 5.0
        - metrics.max_drawdown * 1.5
        + min(metrics.days / TRADING_DAYS_PER_YEAR, 2.0) * 0.05
    )


def build_walk_forward_windows(
    dates: list[date],
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict[str, date | int]]:
    windows: list[dict[str, date | int]] = []
    start = 0
    fold = 1
    while start + train_days + test_days <= len(dates):
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end = dates[start + train_days + test_days - 1]
        windows.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        fold += 1
        start += step_days
    return windows


def walk_forward_candidates(
    selected_variant: StrategyVariant,
    seed: int,
    max_variants: int,
) -> list[StrategyVariant]:
    candidates: list[StrategyVariant] = []
    for variant in variant_stream(seed):
        candidates.append(variant)
        if len(candidates) >= max_variants:
            break
    candidates.append(selected_variant)
    candidates.extend(variant_neighbors(selected_variant))
    return unique_variants(candidates)


def summarize_fold_metrics(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    if not rows:
        return {
            "folds": 0,
            "positive_return_fraction": 0.0,
            "positive_sharpe_fraction": 0.0,
        }
    metrics = [row[f"{prefix}_metrics"] for row in rows]
    sharpes = [item["sharpe"] for item in metrics]
    annual_returns = [item["annual_return"] for item in metrics]
    drawdowns = [item["max_drawdown"] for item in metrics]
    return {
        "folds": len(rows),
        "positive_return_fraction": fraction(value > 0 for value in annual_returns),
        "positive_sharpe_fraction": fraction(value > 0 for value in sharpes),
        "median_sharpe": median(sharpes),
        "mean_sharpe": mean(sharpes),
        "median_annual_return": median(annual_returns),
        "mean_annual_return": mean(annual_returns),
        "worst_drawdown": max(drawdowns),
    }


def run_walk_forward_validation(
    markets: list[MarketData],
    selected: Evaluation,
    config: SearchConfig,
) -> dict[str, object]:
    universes = build_universes(markets)
    symbols = universes[selected.universe]
    market_by_symbol = {market.symbol: market for market in markets}
    selected_markets = [market_by_symbol[symbol] for symbol in symbols]
    all_dates = sorted({one_date for market in selected_markets for one_date in market.dates})
    windows = build_walk_forward_windows(
        all_dates,
        config.walk_forward_train_days,
        config.walk_forward_test_days,
        config.walk_forward_step_days,
    )

    if not windows:
        return {
            "status": "not_run",
            "reason": "not_enough_history_for_requested_walk_forward_windows",
            "folds": [],
        }

    candidate_variants = walk_forward_candidates(
        selected.variant,
        config.seed,
        config.walk_forward_max_variants,
    )
    candidate_runs = {
        variant: backtest_variant(selected_markets, variant, config.cost_bps)
        for variant in candidate_variants
    }
    all_selected_runs = backtest_variant(markets, selected.variant, config.cost_bps)
    selected_runs = candidate_runs[selected.variant]
    rows: list[dict[str, object]] = []

    for window in windows:
        train_start = window["train_start"]
        train_end = window["train_end"]
        test_start = window["test_start"]
        test_end = window["test_end"]

        best_variant: StrategyVariant | None = None
        best_train_metrics: Metrics | None = None
        best_score = -math.inf

        for variant, runs in candidate_runs.items():
            train_metrics, active_train_symbols = metrics_between(
                runs,
                symbols,
                train_start,
                train_end,
            )
            if active_train_symbols < max(1, min(config.min_markets, len(symbols))):
                continue
            score = period_score(train_metrics)
            if score > best_score:
                best_score = score
                best_variant = variant
                best_train_metrics = train_metrics

        if best_variant is None or best_train_metrics is None:
            continue

        optimized_test_metrics, active_test_symbols = metrics_between(
            candidate_runs[best_variant],
            symbols,
            test_start,
            test_end,
        )
        confirmed_test_metrics, _active_confirmed_symbols = metrics_between(
            selected_runs,
            symbols,
            test_start,
            test_end,
        )
        confirmed_train_metrics, _active_selected_train_symbols = metrics_between(
            selected_runs,
            symbols,
            train_start,
            train_end,
        )

        rows.append(
            {
                "fold": window["fold"],
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "active_test_symbols": active_test_symbols,
                "optimized_strategy_label": best_variant.label(),
                "optimized_train_metrics": best_train_metrics.to_dict(),
                "optimized_test_metrics": optimized_test_metrics.to_dict(),
                "confirmed_strategy_label": selected.variant.label(),
                "confirmed_train_metrics": confirmed_train_metrics.to_dict(),
                "confirmed_test_metrics": confirmed_test_metrics.to_dict(),
                "optimized_sharpe_decay": (
                    best_train_metrics.sharpe - optimized_test_metrics.sharpe
                ),
                "confirmed_sharpe_decay": (
                    confirmed_train_metrics.sharpe - confirmed_test_metrics.sharpe
                ),
            }
        )

    optimized_summary = summarize_fold_metrics(rows, "optimized_test")
    confirmed_summary = summarize_fold_metrics(rows, "confirmed_test")
    optimized_train_summary = summarize_fold_metrics(rows, "optimized_train")
    confirmed_train_summary = summarize_fold_metrics(rows, "confirmed_train")

    overfit_warnings: list[str] = []
    if optimized_summary["positive_return_fraction"] < 0.50:
        overfit_warnings.append("optimized_oos_positive_fraction_below_50_percent")
    if optimized_summary.get("median_sharpe", 0.0) <= 0:
        overfit_warnings.append("optimized_oos_median_sharpe_non_positive")
    if (
        optimized_train_summary.get("mean_sharpe", 0.0)
        - optimized_summary.get("mean_sharpe", 0.0)
        > 1.0
    ):
        overfit_warnings.append("optimized_train_to_oos_sharpe_decay_above_1")
    if confirmed_summary["positive_return_fraction"] < 0.50:
        overfit_warnings.append("confirmed_strategy_oos_positive_fraction_below_50_percent")

    cross_universe = run_cross_universe_walk_forward(
        markets,
        universes,
        all_selected_runs,
        selected.variant,
        config,
    )
    cross_good_fraction = fraction(
        item["summary"]["positive_return_fraction"] >= 0.50
        for item in cross_universe
        if item["summary"]["folds"] > 0
    )
    if cross_good_fraction < 0.50:
        overfit_warnings.append("confirmed_strategy_cross_universe_walk_forward_weak")

    return {
        "status": "passed" if not overfit_warnings else "flagged",
        "warnings": overfit_warnings,
        "candidate_count": len(candidate_variants),
        "universe": selected.universe,
        "market_count": len(symbols),
        "train_days": config.walk_forward_train_days,
        "test_days": config.walk_forward_test_days,
        "step_days": config.walk_forward_step_days,
        "optimized_oos_summary": optimized_summary,
        "optimized_train_summary": optimized_train_summary,
        "confirmed_oos_summary": confirmed_summary,
        "confirmed_train_summary": confirmed_train_summary,
        "cross_universe_confirmed_summary": {
            "universe_count": len(cross_universe),
            "good_universe_fraction": cross_good_fraction,
            "universes": cross_universe,
        },
        "folds": rows,
    }


def run_cross_universe_walk_forward(
    markets: list[MarketData],
    universes: dict[str, list[str]],
    all_selected_runs: dict[str, MarketRun],
    variant: StrategyVariant,
    config: SearchConfig,
) -> list[dict[str, object]]:
    market_by_symbol = {market.symbol: market for market in markets}
    summaries: list[dict[str, object]] = []

    for universe, symbols in universes.items():
        if len(symbols) < max(1, min(config.min_markets, len(symbols))):
            continue
        universe_markets = [market_by_symbol[symbol] for symbol in symbols]
        universe_dates = sorted(
            {one_date for market in universe_markets for one_date in market.dates}
        )
        windows = build_walk_forward_windows(
            universe_dates,
            config.walk_forward_train_days,
            config.walk_forward_test_days,
            config.walk_forward_step_days,
        )
        fold_rows: list[dict[str, object]] = []
        minimum_markets = max(1, min(config.min_markets, len(symbols)))

        for window in windows:
            test_metrics, active_test_symbols = metrics_between(
                all_selected_runs,
                symbols,
                window["test_start"],
                window["test_end"],
            )
            if active_test_symbols < minimum_markets:
                continue
            fold_rows.append(
                {
                    "fold": window["fold"],
                    "test_start": window["test_start"].isoformat(),
                    "test_end": window["test_end"].isoformat(),
                    "active_test_symbols": active_test_symbols,
                    "confirmed_test_metrics": test_metrics.to_dict(),
                }
            )

        summaries.append(
            {
                "universe": universe,
                "market_count": len(symbols),
                "strategy_label": variant.label(),
                "summary": summarize_fold_metrics(fold_rows, "confirmed_test"),
                "folds": fold_rows,
            }
        )

    summaries.sort(
        key=lambda item: (
            item["summary"].get("positive_return_fraction", 0.0),
            item["summary"].get("median_sharpe", -math.inf),
        ),
        reverse=True,
    )
    return summaries


def keep_top(
    top: list[Evaluation], candidate: Evaluation, top_count: int
) -> list[Evaluation]:
    top.append(candidate)
    top.sort(key=lambda item: item.score, reverse=True)
    del top[top_count:]
    return top


def format_pct(value: float) -> str:
    return f"{value * 100:7.2f}%"


def print_evaluation(prefix: str, evaluation: Evaluation) -> None:
    print(
        prefix,
        evaluation.universe,
        evaluation.variant.label(),
        "test_ann",
        format_pct(evaluation.test.annual_return),
        "test_sharpe",
        f"{evaluation.test.sharpe:5.2f}",
        "test_dd",
        format_pct(evaluation.test.max_drawdown),
        "val_sharpe",
        f"{evaluation.validation.sharpe:5.2f}",
        "pos_mkts",
        f"{evaluation.positive_test_market_fraction:4.0%}",
    )


def run_search(
    markets: list[MarketData],
    config: SearchConfig,
) -> tuple[Evaluation | None, list[Evaluation], list[Evaluation], dict[str, object]]:
    train_end, validation_end = choose_split_dates(markets)
    universes = build_universes(markets)
    started = time.monotonic()
    best: Evaluation | None = None
    found: Evaluation | None = None
    top: list[Evaluation] = []
    all_evaluations: list[Evaluation] = []
    trials = 0
    variants_tested = 0
    last_progress = started

    print(
        f"Loaded {len(markets)} markets. Split: train <= {train_end}, "
        f"validation <= {validation_end}, test after {validation_end}."
    )
    print(
        f"Searching with {config.cost_bps:g} bps cost, "
        f"{config.max_trials} max universe trials, "
        f"{config.time_budget_sec:g}s time budget."
    )

    for variant in variant_stream(config.seed):
        elapsed = time.monotonic() - started
        if trials >= config.max_trials or elapsed >= config.time_budget_sec:
            break

        runs = backtest_variant(markets, variant, config.cost_bps)
        variants_tested += 1

        for universe, symbols in universes.items():
            if trials >= config.max_trials:
                break
            if time.monotonic() - started >= config.time_budget_sec:
                break

            evaluation = evaluate_universe(
                variant,
                universe,
                symbols,
                runs,
                train_end,
                validation_end,
                config,
            )
            if evaluation is None:
                continue

            trials += 1
            all_evaluations.append(evaluation)
            top = keep_top(top, evaluation, config.top_count)
            if best is None or evaluation.score > best.score:
                best = evaluation
                print_evaluation("New best:", evaluation)

            if evaluation.passed:
                if found is None or evaluation.score > found.score:
                    found = evaluation
                print_evaluation("Research gate passed:", evaluation)
                if config.stop_on_pass:
                    elapsed = time.monotonic() - started
                    return found, top, all_evaluations, {
                        "trials": trials,
                        "variants_tested": variants_tested,
                        "elapsed_sec": elapsed,
                        "train_end": train_end.isoformat(),
                        "validation_end": validation_end.isoformat(),
                        "universes": {k: len(v) for k, v in universes.items()},
                    }

        now = time.monotonic()
        if now - last_progress >= 15:
            last_progress = now
            if best:
                print_evaluation(
                    f"Progress: {trials} trials, {variants_tested} variants. Best:",
                    best,
                )

    elapsed = time.monotonic() - started
    return found, top, all_evaluations, {
        "trials": trials,
        "variants_tested": variants_tested,
        "elapsed_sec": elapsed,
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "universes": {k: len(v) for k, v in universes.items()},
    }


EVALUATION_CSV_HEADER = [
    "rank",
    "passed",
    "score",
    "universe",
    "market_count",
    "strategy",
    "train_sharpe",
    "train_annual_return",
    "train_max_drawdown",
    "validation_sharpe",
    "validation_annual_return",
    "validation_max_drawdown",
    "test_sharpe",
    "test_annual_return",
    "test_total_return",
    "test_max_drawdown",
    "positive_test_market_fraction",
    "total_trades",
    "exposure_fraction",
    "failed_reasons",
]


def evaluation_csv_row(rank: int, item: Evaluation) -> list[object]:
    return [
        rank,
        item.passed,
        round(item.score, 6),
        item.universe,
        item.market_count,
        item.variant.label(),
        round(item.train.sharpe, 6),
        round(item.train.annual_return, 6),
        round(item.train.max_drawdown, 6),
        round(item.validation.sharpe, 6),
        round(item.validation.annual_return, 6),
        round(item.validation.max_drawdown, 6),
        round(item.test.sharpe, 6),
        round(item.test.annual_return, 6),
        round(item.test.total_return, 6),
        round(item.test.max_drawdown, 6),
        round(item.positive_test_market_fraction, 6),
        item.total_trades,
        round(item.exposure_fraction, 6),
        ";".join(item.failed_reasons),
    ]


def write_evaluations_csv(path: Path, evaluations: list[Evaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVALUATION_CSV_HEADER)
        for rank, item in enumerate(evaluations, start=1):
            writer.writerow(evaluation_csv_row(rank, item))


def evaluation_dict_csv_row(
    check_type: str, item_name: str, evaluation: dict[str, object]
) -> list[object]:
    train = evaluation["train"]
    validation = evaluation["validation"]
    test = evaluation["test"]
    return [
        check_type,
        item_name,
        evaluation["passed"],
        round(float(evaluation["score"]), 6),
        evaluation["universe"],
        evaluation["market_count"],
        evaluation["strategy_label"],
        round(float(train["sharpe"]), 6),
        round(float(train["annual_return"]), 6),
        round(float(validation["sharpe"]), 6),
        round(float(validation["annual_return"]), 6),
        round(float(test["sharpe"]), 6),
        round(float(test["annual_return"]), 6),
        round(float(test["total_return"]), 6),
        round(float(test["max_drawdown"]), 6),
        round(float(evaluation["positive_test_market_fraction"]), 6),
        evaluation["total_trades"],
        round(float(evaluation["exposure_fraction"]), 6),
        ";".join(evaluation["failed_reasons"]),
    ]


def write_robustness_csv(path: Path, report: dict[str, object] | None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "check_type",
                "item",
                "passed",
                "score",
                "universe",
                "market_count",
                "strategy",
                "train_sharpe",
                "train_annual_return",
                "validation_sharpe",
                "validation_annual_return",
                "test_sharpe",
                "test_annual_return",
                "test_total_return",
                "test_max_drawdown",
                "positive_test_market_fraction",
                "total_trades",
                "exposure_fraction",
                "failed_reasons",
            ]
        )
        if report is None:
            return
        for item in report.get("cost_tests", []):
            writer.writerow(
                evaluation_dict_csv_row(
                    "cost_stress",
                    f"{item['cost_bps']}bps",
                    item["evaluation"],
                )
            )
        for item in report.get("parameter_neighbors", []):
            writer.writerow(evaluation_dict_csv_row("parameter_neighbor", "", item))
        for item in report.get("leave_one_out", []):
            writer.writerow(
                evaluation_dict_csv_row(
                    "leave_one_out",
                    str(item["omitted_symbol"]),
                    item["evaluation"],
                )
            )
        for item in report.get("market_breakdown", []):
            writer.writerow(evaluation_dict_csv_row("single_market", "", item))


def write_walk_forward_csv(path: Path, report: dict[str, object] | None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scope",
                "universe",
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "active_test_symbols",
                "optimized_strategy",
                "optimized_train_sharpe",
                "optimized_train_annual_return",
                "optimized_test_sharpe",
                "optimized_test_annual_return",
                "optimized_test_max_drawdown",
                "confirmed_strategy",
                "confirmed_train_sharpe",
                "confirmed_train_annual_return",
                "confirmed_test_sharpe",
                "confirmed_test_annual_return",
                "confirmed_test_max_drawdown",
                "optimized_sharpe_decay",
                "confirmed_sharpe_decay",
            ]
        )
        if report is None:
            return
        for row in report.get("folds", []):
            optimized_train = row["optimized_train_metrics"]
            optimized_test = row["optimized_test_metrics"]
            confirmed_train = row["confirmed_train_metrics"]
            confirmed_test = row["confirmed_test_metrics"]
            writer.writerow(
                [
                    "selected_universe_optimized",
                    report.get("universe", ""),
                    row["fold"],
                    row["train_start"],
                    row["train_end"],
                    row["test_start"],
                    row["test_end"],
                    row["active_test_symbols"],
                    row["optimized_strategy_label"],
                    round(float(optimized_train["sharpe"]), 6),
                    round(float(optimized_train["annual_return"]), 6),
                    round(float(optimized_test["sharpe"]), 6),
                    round(float(optimized_test["annual_return"]), 6),
                    round(float(optimized_test["max_drawdown"]), 6),
                    row["confirmed_strategy_label"],
                    round(float(confirmed_train["sharpe"]), 6),
                    round(float(confirmed_train["annual_return"]), 6),
                    round(float(confirmed_test["sharpe"]), 6),
                    round(float(confirmed_test["annual_return"]), 6),
                    round(float(confirmed_test["max_drawdown"]), 6),
                    round(float(row["optimized_sharpe_decay"]), 6),
                    round(float(row["confirmed_sharpe_decay"]), 6),
                ]
            )
        cross_universe = report.get("cross_universe_confirmed_summary", {})
        for universe_report in cross_universe.get("universes", []):
            for row in universe_report.get("folds", []):
                confirmed_test = row["confirmed_test_metrics"]
                writer.writerow(
                    [
                        "cross_universe_confirmed",
                        universe_report["universe"],
                        row["fold"],
                        "",
                        "",
                        row["test_start"],
                        row["test_end"],
                        row["active_test_symbols"],
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        universe_report["strategy_label"],
                        "",
                        "",
                        round(float(confirmed_test["sharpe"]), 6),
                        round(float(confirmed_test["annual_return"]), 6),
                        round(float(confirmed_test["max_drawdown"]), 6),
                        "",
                        "",
                    ]
                )


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_validated_json(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    json.loads(text)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_reports(
    results_dir: Path,
    found: Evaluation | None,
    top: list[Evaluation],
    all_evaluations: list[Evaluation],
    metadata: dict[str, object],
    config: SearchConfig,
    robustness: dict[str, object] | None = None,
    walk_forward: dict[str, object] | None = None,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"strategy_search_{stamp}.json"
    csv_path = results_dir / f"strategy_search_{stamp}.csv"
    all_csv_path = results_dir / f"strategy_search_{stamp}_all.csv"
    failed_csv_path = results_dir / f"strategy_search_{stamp}_failed.csv"
    robustness_csv_path = results_dir / f"strategy_search_{stamp}_robustness.csv"
    walk_forward_csv_path = results_dir / f"strategy_search_{stamp}_walk_forward.csv"

    report = {
        "status": "passed" if found else "not_found",
        "found": found.to_dict() if found else None,
        "top": [item.to_dict() for item in top],
        "metadata": metadata,
        "robustness": robustness,
        "walk_forward": walk_forward,
        "log_files": {
            "top_csv": str(csv_path),
            "all_csv": str(all_csv_path),
            "failed_csv": str(failed_csv_path),
            "robustness_csv": str(robustness_csv_path),
            "walk_forward_csv": str(walk_forward_csv_path),
        },
        "criteria": {
            "min_oos_days": config.min_oos_days,
            "min_total_trades": config.min_total_trades,
            "min_exposure_fraction": config.min_exposure_fraction,
            "target_test_sharpe": config.target_test_sharpe,
            "target_test_annual_return": config.target_test_annual_return,
            "max_test_drawdown": config.max_test_drawdown,
            "min_validation_sharpe": config.min_validation_sharpe,
            "min_train_sharpe": config.min_train_sharpe,
            "min_positive_market_fraction": config.min_positive_market_fraction,
            "cost_bps": config.cost_bps,
        },
    }
    write_validated_json(json_path, report)

    failed_evaluations = [item for item in all_evaluations if not item.passed]
    write_evaluations_csv(csv_path, top)
    write_evaluations_csv(all_csv_path, all_evaluations)
    write_evaluations_csv(failed_csv_path, failed_evaluations)
    write_robustness_csv(robustness_csv_path, robustness)
    write_walk_forward_csv(walk_forward_csv_path, walk_forward)

    latest_json = results_dir / "latest_strategy_search.json"
    latest_csv = results_dir / "latest_strategy_search.csv"
    latest_all_csv = results_dir / "latest_strategy_search_all.csv"
    latest_failed_csv = results_dir / "latest_strategy_search_failed.csv"
    latest_robustness_csv = results_dir / "latest_strategy_search_robustness.csv"
    latest_walk_forward_csv = results_dir / "latest_strategy_search_walk_forward.csv"
    write_validated_json(latest_json, report)
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_all_csv.write_text(all_csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_failed_csv.write_text(
        failed_csv_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    latest_robustness_csv.write_text(
        robustness_csv_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    latest_walk_forward_csv.write_text(
        walk_forward_csv_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return {
        "json": json_path,
        "top_csv": csv_path,
        "all_csv": all_csv_path,
        "failed_csv": failed_csv_path,
        "robustness_csv": robustness_csv_path,
        "walk_forward_csv": walk_forward_csv_path,
    }


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Multipliers must be positive")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search daily OHLCV futures data for historically robust strategy "
            "candidates. This is research tooling, not financial advice."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--max-trials", type=int, default=5000)
    parser.add_argument("--time-budget-sec", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-searching-after-pass", action="store_true")
    parser.add_argument("--min-bars", type=int, default=750)
    parser.add_argument("--min-markets", type=int, default=4)
    parser.add_argument("--min-oos-days", type=int, default=750)
    parser.add_argument("--min-total-trades", type=int, default=100)
    parser.add_argument("--min-exposure-fraction", type=float, default=0.08)
    parser.add_argument("--target-test-sharpe", type=float, default=0.75)
    parser.add_argument("--target-test-annual-return", type=float, default=0.05)
    parser.add_argument("--max-test-drawdown", type=float, default=0.25)
    parser.add_argument("--min-validation-sharpe", type=float, default=0.10)
    parser.add_argument("--min-train-sharpe", type=float, default=0.10)
    parser.add_argument("--min-positive-market-fraction", type=float, default=0.50)
    parser.add_argument("--top-count", type=int, default=25)
    parser.add_argument(
        "--skip-post-confirmation-checks",
        action="store_true",
        help="Skip robustness and walk-forward checks after a profitable candidate is found.",
    )
    parser.add_argument(
        "--robustness-cost-multipliers",
        type=parse_float_tuple,
        default=(1.0, 2.0, 3.0, 5.0),
        help="Comma-separated transaction-cost multipliers for robustness checks.",
    )
    parser.add_argument("--walk-forward-train-days", type=int, default=1260)
    parser.add_argument("--walk-forward-test-days", type=int, default=252)
    parser.add_argument("--walk-forward-step-days", type=int, default=252)
    parser.add_argument("--walk-forward-max-variants", type=int, default=250)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        cost_bps=args.cost_bps,
        max_trials=args.max_trials,
        time_budget_sec=args.time_budget_sec,
        seed=args.seed,
        stop_on_pass=not args.keep_searching_after_pass,
        min_bars=args.min_bars,
        min_markets=args.min_markets,
        min_oos_days=args.min_oos_days,
        min_total_trades=args.min_total_trades,
        min_exposure_fraction=args.min_exposure_fraction,
        target_test_sharpe=args.target_test_sharpe,
        target_test_annual_return=args.target_test_annual_return,
        max_test_drawdown=args.max_test_drawdown,
        min_validation_sharpe=args.min_validation_sharpe,
        min_train_sharpe=args.min_train_sharpe,
        min_positive_market_fraction=args.min_positive_market_fraction,
        top_count=args.top_count,
        run_post_confirmation_checks=not args.skip_post_confirmation_checks,
        robustness_cost_multipliers=args.robustness_cost_multipliers,
        walk_forward_train_days=args.walk_forward_train_days,
        walk_forward_test_days=args.walk_forward_test_days,
        walk_forward_step_days=args.walk_forward_step_days,
        walk_forward_max_variants=args.walk_forward_max_variants,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    markets = load_markets(config.data_dir, config.min_bars)
    found, top, all_evaluations, metadata = run_search(markets, config)
    robustness: dict[str, object] | None = None
    walk_forward: dict[str, object] | None = None

    if found and config.run_post_confirmation_checks:
        print()
        print("Running robustness checks on selected candidate.")
        robustness = run_robustness_tests(markets, found, config)
        print(
            "Robustness status:",
            robustness["status"],
            "warnings:",
            ", ".join(robustness["warnings"]) or "none",
        )
        print("Running walk-forward validation and overfit diagnostics.")
        walk_forward = run_walk_forward_validation(markets, found, config)
        print(
            "Walk-forward status:",
            walk_forward["status"],
            "warnings:",
            ", ".join(walk_forward.get("warnings", [])) or "none",
        )

    paths = write_reports(
        config.results_dir,
        found,
        top,
        all_evaluations,
        metadata,
        config,
        robustness,
        walk_forward,
    )

    print()
    if found:
        print_evaluation("Selected candidate:", found)
    else:
        print("No candidate passed the research gate within the configured budget.")
        if top:
            print_evaluation("Best candidate:", top[0])
    print(f"JSON report: {paths['json']}")
    print(f"Top CSV:     {paths['top_csv']}")
    print(f"All CSV:     {paths['all_csv']}")
    print(f"Failed CSV:  {paths['failed_csv']}")
    print(f"Robust CSV:  {paths['robustness_csv']}")
    print(f"Walk CSV:    {paths['walk_forward_csv']}")
    print("Treat any result as historical research, not a live-trading guarantee.")
    return 0
