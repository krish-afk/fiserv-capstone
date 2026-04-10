# Trading Strategies

This folder supports **plug-in trading strategies**.

A team member can add a new strategy file under `src/trading/strategies/`, set the file/class in `config.yaml`, and run it through the same trading pipeline as `krish_trade`.

## Folder structure

```text
src/
  trading/
    __init__.py
    strategy.py
    backtest.py
    pipeline.py
    performance.py
    strategies/
      __init__.py
      krish_trade_strategy.py
      skeleton_strategy.py
      your_strategy.py
```

## What the pipeline does

When the trading pipeline runs, it will:

1. Load the strategy file from `config.yaml`
2. Instantiate the strategy class from that file
3. Load the required forecast panels
4. Pull market data for any tickers declared by the strategy
5. Run the backtest using the same monthly decision/rebalance flow as `krish_trade`
6. Save outputs into the experiment run folder under `trading/`

## Team workflow

### 1) Copy the skeleton file

Start from:

```python
src/trading/strategies/skeleton_strategy.py
```

Copy it to a new file, for example:

```text
src/trading/strategies/alex_rotation_strategy.py
```

### 2) Rename the class

Inside the file:

```python
@register_strategy
class AlexRotationStrategy(BaseStrategy):
    ...
```

### 3) Set the strategy in `config.yaml`

```yaml
trading:
  strategy:
    file: alex_rotation_strategy.py
    class: AlexRotationStrategy
    params:
      risk_on_ticker: SPY
      defensive_ticker: TLT
      threshold: 0.15
      target_allocation: 0.50

  forecast_panels:
    primary: pce
    mrts: mrts

  market_data:
    tickers: []
    start_date: 2018-01-01
    end_date: 2026-01-01
    source: yfinance
```

That is all the pipeline needs to load and run your strategy.

---

## Required strategy structure

Each strategy file should define **one main strategy class** that inherits from `BaseStrategy`.

Minimal example:

```python
from src.trading.strategy import BaseStrategy, StrategyData, register_strategy

@register_strategy
class MyStrategy(BaseStrategy):
    def __init__(self, threshold: float = 0.1, **params):
        super().__init__(threshold=threshold, **params)
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "my_strategy"

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts"}

    @property
    def tickers(self) -> list[str]:
        return ["SPY", "TLT"]

    def generate_signals(self, data: StrategyData):
        ...
```

---

## What each part means

### `name`
Readable strategy name used in output filenames and result metadata.

### `required_inputs`
Set of data inputs your strategy needs.

Available values:

- `"forecasts"` – primary forecast panel
- `"prices"` – market OHLCV data
- `"macro"` – macro data frame
- `"mrts"` – MRTS forecast panel

Examples:

```python
return {"forecasts"}
```

```python
return {"forecasts", "prices"}
```

```python
return {"forecasts", "mrts", "prices"}
```

### `tickers`
List of tickers the strategy needs market data for.

Example:

```python
return ["SPY", "TLT"]
```

The pipeline automatically includes these tickers when pulling market data.

### `generate_signals(data)`
This is where your trading logic lives.

The method receives a `StrategyData` object with:

- `data.forecasts`
- `data.prices`
- `data.macro`
- `data.mrts`
- `data.cfg`
- `data.context`

Use `data.validate(self.required_inputs)` at the start to ensure required data exists.

---

## Strategy output format

Preferred output: a DataFrame indexed by `date` with **weight columns**.

### Preferred format

```text
weight__SPY
weight__TLT
cash_weight
confidence
metadata
```

Example:

```python
return self._make_weight_frame(
    index=forecasts.index,
    weights={
        "SPY": spy_weight,
        "TLT": tlt_weight,
    },
    cash_weight=cash_weight,
    confidence=confidence,
    metadata=metadata,
)
```

### Legacy format
The backtest still accepts the old format:

```text
signal
confidence
metadata
```

But new strategies should use `weight__TICKER` columns.

---

## Accessing forecast data

Most strategies will start like this:

```python
forecasts = self._coerce_forecast_index(data.forecasts)
score = forecasts["y_pred"].astype(float)
```

That gives you a date-indexed frame to build your signals from.

---

## Accessing market data

If your strategy needs prices, add `"prices"` to `required_inputs` and declare the tickers in `tickers`.

Example:

```python
@property
def required_inputs(self) -> set[str]:
    return {"forecasts", "prices"}

@property
def tickers(self) -> list[str]:
    return ["SPY", "TLT"]
```

Then inside `generate_signals()`:

```python
spy_close = self.get_price_series(data.prices, "SPY", "close")
tlt_close = self.get_price_series(data.prices, "TLT", "close")
```

Helper:

```python
self.get_price_series(data.prices, ticker, field="close")
```

Supported fields depend on what exists in the pulled market data, typically:

- `open`
- `high`
- `low`
- `close`
- `volume`

Column names are normalized internally, so strategies can safely ask for `"SPY"` and `"close"` even if raw storage used lowercase column names like `spy_close`.

---

## Accessing MRTS forecasts

If your strategy needs MRTS as well as the primary panel:

```python
@property
def required_inputs(self) -> set[str]:
    return {"forecasts", "mrts"}
```

Then:

```python
pce = self._coerce_forecast_index(data.forecasts)
mrts = self._coerce_forecast_index(data.mrts)
```

This is how `krish_trade_strategy.py` combines PCE and MRTS signals.

---

## Accessing config values

The full loaded config is available at:

```python
data.config
```

or:

```python
data.cfg
```

You can also use constructor params from `config.yaml` under:

```yaml
trading:
  strategy:
    params:
      ...
```

Those are passed into your strategy `__init__()` automatically.

---

## Example pattern for a custom strategy

```python
from __future__ import annotations

import numpy as np
from src.trading.strategy import BaseStrategy, StrategyData, register_strategy


@register_strategy
class ExampleMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        risk_on_ticker: str = "SPY",
        defensive_ticker: str = "TLT",
        threshold: float = 0.1,
        target_allocation: float = 0.5,
        **params,
    ):
        super().__init__(
            risk_on_ticker=risk_on_ticker,
            defensive_ticker=defensive_ticker,
            threshold=threshold,
            target_allocation=target_allocation,
            **params,
        )
        self.risk_on_ticker = risk_on_ticker.upper()
        self.defensive_ticker = defensive_ticker.upper()
        self.threshold = float(threshold)
        self.target_allocation = float(target_allocation)

    @property
    def name(self) -> str:
        return "example_momentum"

    @property
    def required_inputs(self) -> set[str]:
        return {"forecasts"}

    @property
    def tickers(self) -> list[str]:
        return [self.risk_on_ticker, self.defensive_ticker]

    def generate_signals(self, data: StrategyData):
        data.validate(self.required_inputs)
        forecasts = self._coerce_forecast_index(data.forecasts)
        score = forecasts["y_pred"].astype(float)

        risk_on_weight = np.where(score > self.threshold, self.target_allocation, 0.0)
        defensive_weight = np.where(score <= self.threshold, self.target_allocation, 0.0)
        cash_weight = 1.0 - risk_on_weight - defensive_weight
        confidence = np.clip(np.abs(score), 0.0, 1.0)

        metadata = [{"forecast": float(v)} for v in score]

        return self._make_weight_frame(
            index=forecasts.index,
            weights={
                self.risk_on_ticker: risk_on_weight,
                self.defensive_ticker: defensive_weight,
            },
            cash_weight=cash_weight,
            confidence=confidence,
            metadata=metadata,
        )
```

---

## Running a strategy

Once `config.yaml` points at the strategy, the trading pipeline can be run through:

```python
from src.trading.pipeline import run_configured_trading_pipeline

result = run_configured_trading_pipeline()
```

This will:

- load the configured strategy
- build the required data inputs
- fetch/load market data for declared tickers
- run the backtest
- write outputs to the latest experiment run folder under `trading/`

---

## Output files

The trading pipeline writes files like:

```text
signals_<strategy>.csv
positions_<strategy>.csv
backtest_trades_<strategy>.csv
backtest_results_<strategy>.json
equity_curve_<strategy>.csv
```

These are stored in the experiment output folder, typically:

```text
experiments/<timestamp>_run/trading/
```

---

## Rules for team members

### Do
- inherit from `BaseStrategy`
- decorate your strategy with `@register_strategy`
- keep one main strategy class per file
- declare every traded ticker in `tickers`
- use `required_inputs` correctly
- return `weight__TICKER` columns for new strategies

### Don’t
- hardcode file paths inside the strategy
- fetch market data directly inside the strategy
- run backtests inside the strategy file
- rely on global mutable state
- assume price column casing

---

## Common mistakes

### Strategy file not found
Check:

```yaml
trading:
  strategy:
    file: your_strategy.py
```

Make sure the file exists under:

```text
src/trading/strategies/
```

### Strategy class not found
Check:

```yaml
trading:
  strategy:
    class: YourStrategyClass
```

Make sure the class name matches exactly.

### No market data for ticker
Make sure the ticker is returned in:

```python
@property
def tickers(self) -> list[str]:
    return [...]
```

The pipeline only auto-fetches tickers it knows about.

### Strategy needs prices but `data.prices` is None
Make sure `"prices"` is included in `required_inputs`.

### Multiple strategy classes in one file
If you put more than one `BaseStrategy` subclass in the same file, set the exact class in `config.yaml`.

---

## Recommended development flow

1. Copy `skeleton_strategy.py`
2. Rename the file/class
3. Add params to `__init__()`
4. Declare `required_inputs`
5. Declare `tickers`
6. Write `generate_signals()`
7. Point `config.yaml` at the new file/class
8. Run the pipeline
9. Inspect outputs in the experiment run folder

---

## Starter files

Use these as references:

- `src/trading/strategies/krish_trade_strategy.py`
- `src/trading/strategies/skeleton_strategy.py`

`krish_trade_strategy.py` shows a real strategy with multiple forecast panels.

`skeleton_strategy.py` is the cleanest template for team members to copy.

---

## Summary

To add a strategy, a team member only needs to:

1. create a file in `src/trading/strategies/`
2. inherit from `BaseStrategy`
3. declare `tickers` and `required_inputs`
4. implement `generate_signals()`
5. set `trading.strategy.file` and `trading.strategy.class` in `config.yaml`

After that, the same trading pipeline will pick it up and run it.
