# StrategyLooper

Self-contained research harness for daily OHLCV `.dat` files in `data/`.

The tool reads the binary OHLCV format directly with the Python standard library,
generates strategy candidates, backtests them without lookahead by applying each
signal to the next close-to-close return, charges configurable transaction costs,
and checks train/validation/test robustness gates.

This is historical research tooling, not financial advice or a guarantee of live
profitability.

## Run

```powershell
python main.py
```

By default, the loop stops at the first candidate that passes the research gate.
To keep exploring after the first pass:

```powershell
python main.py --max-trials 1200 --time-budget-sec 90 --keep-searching-after-pass
```

Reports are written to `results/` as JSON and CSV files.

Each run writes:

- `latest_strategy_search.csv`: top-ranked candidates
- `latest_strategy_search_all.csv`: every evaluated candidate, including failures
- `latest_strategy_search_failed.csv`: failed candidates with rejection reasons
- `latest_strategy_search_robustness.csv`: cost stress, parameter-neighbor,
  leave-one-market-out, and single-market checks for the selected candidate
- `latest_strategy_search_walk_forward.csv`: walk-forward folds for the selected
  universe plus cross-universe fixed-strategy checks across the available files

## Default Research Gate

- 2 bps transaction cost per position change
- train split through 2016-12-19
- validation split through 2021-08-18
- test split after 2021-08-18
- at least 750 out-of-sample days
- at least 100 trades
- at least 8% exposure
- train Sharpe >= 0.10
- validation Sharpe >= 0.10 and positive annual return
- test Sharpe >= 0.75
- test annual return >= 5%
- test max drawdown <= 25%
- at least 50% of markets positive in the test split

## Post-Confirmation Checks

When a candidate passes the default research gate, the tool automatically runs:

- cost stress tests at 1x, 2x, 3x, and 5x the configured transaction cost
- nearby-parameter tests to see whether the edge survives small parameter changes
- leave-one-market-out tests to detect dependence on one contract
- single-market breakdowns
- rolling walk-forward validation using a 5-year train window and 1-year test
  window
- cross-universe walk-forward checks that apply the confirmed strategy across all
  predefined universes

Use `--skip-post-confirmation-checks` for fast smoke tests.

## Notebook Review

Open `StrategyLooper_Review.ipynb` in Jupyter after a run to review charts and
tables. The notebook reads the `results/latest_*` files and rebuilds the selected
strategy from the raw `.dat` files.

The notebook includes:

- selected strategy summary
- top strategy table
- failed-reason counts
- equity curve
- drawdown curve
- train/validation/test metric charts
- cost-stress tables
- parameter-neighbor charts
- leave-one-market-out checks
- single-market breakdowns
- walk-forward return, Sharpe, and drawdown charts
- cross-universe walk-forward charts

If you want the notebook to regenerate results, uncomment the optional search
cell near the top.
