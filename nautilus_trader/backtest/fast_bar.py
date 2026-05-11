# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Fast candle/bar backtest helpers for constrained one-bar limit strategies.

This module intentionally does not replace the event-driven ``BacktestEngine``.
It provides a Nautilus-owned execution path for research workloads where the
strategy state can be represented as per-bar target exposure arrays and each
instrument has at most one resting limit order that expires on the next bar.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class CausalityError(ValueError):
    """
    Raised when a fast-bar input could expose data after the decision timestamp.
    """


@dataclass(frozen=True, slots=True)
class CausalArray:
    """
    One-dimensional feature values with explicit availability metadata.

    ``value_time`` is the timestamp the output row is aligned to. ``available_at``
    is the earliest timestamp at which the value may be consumed by user logic.
    Fast backtests validate ``available_at <= decision_time`` before execution.
    """

    values: FloatArray
    value_time: IntArray
    available_at: IntArray | None = None
    name: str = ""
    lineage: tuple[str, ...] = ()
    live_compatible: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        value_time = np.asarray(self.value_time, dtype=np.int64)
        available_at = (
            value_time
            if self.available_at is None
            else np.asarray(self.available_at, dtype=np.int64)
        )

        if values.ndim != 1:
            raise ValueError(f"{self._label()} values must be a one-dimensional array")
        if value_time.ndim != 1:
            raise ValueError(f"{self._label()} value_time must be a one-dimensional array")
        if available_at.ndim != 1:
            raise ValueError(f"{self._label()} available_at must be a one-dimensional array")
        if len(value_time) != len(values) or len(available_at) != len(values):
            raise ValueError(
                f"{self._label()} values, value_time, and available_at must have the same length",
            )

        # Metadata must stay stable after validation; values remain zero-copy for speed.
        value_time.setflags(write=False)
        available_at.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_time", value_time)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "lineage", tuple(self.lineage))

    def validate_available_at_or_before(
        self,
        decision_time: IntArray | int,
        *,
        name: str | None = None,
    ) -> None:
        """
        Reject any row whose feature was not available at the decision timestamp.
        """
        decisions = np.asarray(decision_time, dtype=np.int64)
        if decisions.ndim == 0:
            decisions = np.full(len(self.values), int(decisions), dtype=np.int64)
        if decisions.ndim != 1:
            raise ValueError("decision_time must be a scalar or one-dimensional array")
        if len(decisions) != len(self.values):
            raise ValueError("decision_time must have the same length as feature values")

        future_mask = self.available_at > decisions
        if future_mask.any():
            index = int(np.flatnonzero(future_mask)[0])
            field_name = name or self._label()
            raise CausalityError(
                f"{field_name} is available after the decision timestamp at index {index}: "
                f"available_at={int(self.available_at[index])}, "
                f"decision_time={int(decisions[index])}",
            )

    def _label(self) -> str:
        return self.name or "CausalArray"


@dataclass(frozen=True, slots=True)
class CausalFrame:
    """
    Named causal arrays produced by a feature pipeline for one instrument.
    """

    columns: Mapping[str, CausalArray]

    def __post_init__(self) -> None:
        columns = dict(self.columns)
        for name, column in columns.items():
            if not isinstance(column, CausalArray):
                raise TypeError(f"CausalFrame column {name!r} must be a CausalArray")
        object.__setattr__(self, "columns", columns)

    def require(self, *names: str) -> tuple[CausalArray, ...]:
        missing = [name for name in names if name not in self.columns]
        if missing:
            raise ValueError(f"CausalFrame missing required columns: {missing}")
        return tuple(self.columns[name] for name in names)

    def validate_available_at_or_before(self, decision_time: IntArray | int) -> None:
        for name, column in self.columns.items():
            column.validate_available_at_or_before(decision_time, name=name)


@dataclass(frozen=True, slots=True)
class TargetExposure:
    """
    Live/batch target emitted by a causal signal model.
    """

    instrument_id: Any
    target_weight: float
    decision_time: int
    available_at: int
    atr: float | None = None


class CausalFeaturePipeline(Protocol):
    """
    User-owned feature/signal contract shared by fast backtests and live bars.
    """

    def warmup_bars(self) -> int:
        """
        Return the minimum number of completed bars needed before valid targets.
        """
        ...

    def is_live_compatible(self) -> bool:
        """
        Return ``True`` only when batch logic has an equivalent online implementation.
        """
        ...

    def compute_batch(self, bars_by_instrument: Mapping[Any, Any]) -> Mapping[Any, CausalFrame]:
        """
        Compute causal batch features for fast backtests.
        """
        ...

    def update_bar_batch(
        self,
        timestamp: int,
        bars: Mapping[Any, Any],
    ) -> Mapping[Any, TargetExposure]:
        """
        Consume one completed live bar batch and emit current targets.
        """
        ...


CausalSignalModel = CausalFeaturePipeline
ArrayInput = FloatArray | CausalArray


@dataclass(frozen=True, slots=True)
class OneBarLimitConfig:
    """
    Configuration for one-bar limit exposure backtests.

    Parameters
    ----------
    bet_size : float
        Notional exposure represented by a target weight of ``1.0``.
    limit_entry_atr_multiple : float
        ATR fraction used to offset buy limits below lows and sell limits above highs.
    min_dollar_volume : float, default 0.0
        Bars below this dollar-volume threshold target zero exposure.
    fee_rate : float, default 0.0
        Proportional fee applied to traded notional.
    strict_crossing : bool, default True
        If ``True``, buy limits fill only when ``low < limit`` and sell limits only
        when ``high > limit``. This matches the MR3 notebook.
    price_precision : int, optional
        Decimal places used to round submitted limit prices. Leave ``None`` to keep
        raw floating-point prices.
    quantity_precision : int, optional
        Decimal places used to round submitted quantities. Leave ``None`` to keep
        raw floating-point quantities.
    min_quantity : float, default 0.0
        Minimum rounded quantity accepted by the fast executor. This is useful for
        mirroring instrument order factories that reject quantities rounded to zero.
    unsafe_assume_causal : bool, default False
        Allow raw NumPy arrays by stamping each row as available at its own row
        timestamp. This is intended only for migration and tests.
    require_live_compatible : bool, default True
        Reject causal arrays that explicitly declare they cannot be reproduced in
        live bar trading.

    """

    bet_size: float
    limit_entry_atr_multiple: float
    min_dollar_volume: float = 0.0
    fee_rate: float = 0.0
    strict_crossing: bool = True
    price_precision: int | None = None
    quantity_precision: int | None = None
    min_quantity: float = 0.0
    unsafe_assume_causal: bool = False
    require_live_compatible: bool = True


@dataclass(frozen=True, slots=True)
class OneBarLimitMarketData:
    """
    Per-instrument arrays consumed by the fast one-bar limit executor.
    """

    label: str
    high: ArrayInput
    low: ArrayInput
    close: ArrayInput
    target_weight: ArrayInput
    atr: ArrayInput
    dollar_volume: ArrayInput | None = None


@dataclass(frozen=True, slots=True)
class OneBarLimitFill:
    """
    Minimal fill record produced by the constrained bar executor.
    """

    label: str
    bar_index: int
    side: str
    price: float
    quantity: float
    notional: float
    position_after: float
    cash_after: float


@dataclass(frozen=True, slots=True)
class OneBarLimitResult:
    """
    Result for one instrument from the constrained bar executor.
    """

    label: str
    positions: FloatArray
    equities: FloatArray
    order_count: int
    fill_count: int
    buy_order_count: int
    sell_order_count: int
    buy_fill_count: int
    sell_fill_count: int
    final_position: float
    final_cash: float
    fills: tuple[OneBarLimitFill, ...]

    @property
    def final_equity(self) -> float:
        """
        Return the last marked equity for the instrument.
        """
        if self.equities.size == 0:
            return self.final_cash
        return float(self.equities[-1])


def simulate_one_bar_limit_market(
    market: OneBarLimitMarketData,
    config: OneBarLimitConfig,
    *,
    record_fills: bool = True,
) -> OneBarLimitResult:
    """
    Simulate one instrument with MR3-style one-bar limit semantics.

    The loop order is:

    1. Fill the prior bar's pending buy/sell limit if the current bar crosses it.
    2. Expire the pending order.
    3. Mark current exposure and place the next one-bar limit from target exposure.

    This deliberately avoids Nautilus command routing, synthetic OHLC tick generation,
    cache mutation, and event publication. Use it only when those event-driven side
    effects are not part of the strategy being tested.
    """
    high_array = _as_causal_array(market.high, "high", config)
    low_array = _as_causal_array(market.low, "low", config)
    close_array = _as_causal_array(market.close, "close", config)
    target_weight_array = _as_causal_array(market.target_weight, "target_weight", config)
    atr_array = _as_causal_array(market.atr, "atr", config)
    dollar_volume_array = (
        CausalArray(
            values=np.full(close_array.values.shape, np.inf, dtype=np.float64),
            value_time=close_array.value_time,
            available_at=close_array.available_at,
            name="dollar_volume",
        )
        if market.dollar_volume is None
        else _as_causal_array(market.dollar_volume, "dollar_volume", config)
    )
    _check_equal_length(
        high_array.values,
        low_array.values,
        close_array.values,
        target_weight_array.values,
        atr_array.values,
        dollar_volume_array.values,
    )
    _validate_market_causality(
        market.label,
        close_array.value_time,
        (
            ("high", high_array),
            ("low", low_array),
            ("close", close_array),
            ("target_weight", target_weight_array),
            ("atr", atr_array),
            ("dollar_volume", dollar_volume_array),
        ),
        config,
    )

    high = high_array.values
    low = low_array.values
    close = close_array.values
    target_weight = target_weight_array.values
    atr = atr_array.values
    dollar_volume = dollar_volume_array.values

    size = len(close)
    positions = np.empty(size, dtype=np.float64)
    equities = np.empty(size, dtype=np.float64)
    fills: list[OneBarLimitFill] = []

    position = 0.0
    cash = 0.0
    buy_price = 0.0
    buy_size = 0.0
    sell_price = np.inf
    sell_size = 0.0

    order_count = 0
    fill_count = 0
    buy_order_count = 0
    sell_order_count = 0
    buy_fill_count = 0
    sell_fill_count = 0

    for i in range(size):
        bar_high = high[i]
        bar_low = low[i]
        bar_close = close[i]
        wanted_weight = target_weight[i]
        bar_atr = atr[i]

        if buy_size > 0.0 and _buy_crossed(bar_low, buy_price, config.strict_crossing):
            traded = buy_size * buy_price
            position += buy_size
            cash -= traded + traded * config.fee_rate
            fill_count += 1
            buy_fill_count += 1
            if record_fills:
                fills.append(
                    OneBarLimitFill(
                        label=market.label,
                        bar_index=i,
                        side="BUY",
                        price=buy_price,
                        quantity=buy_size,
                        notional=traded,
                        position_after=position,
                        cash_after=cash,
                    ),
                )

        if sell_size > 0.0 and _sell_crossed(bar_high, sell_price, config.strict_crossing):
            traded = sell_size * sell_price
            position -= sell_size
            cash += traded - traded * config.fee_rate
            fill_count += 1
            sell_fill_count += 1
            if record_fills:
                fills.append(
                    OneBarLimitFill(
                        label=market.label,
                        bar_index=i,
                        side="SELL",
                        price=sell_price,
                        quantity=sell_size,
                        notional=traded,
                        position_after=position,
                        cash_after=cash,
                    ),
                )

        buy_price = 0.0
        buy_size = 0.0
        sell_price = np.inf
        sell_size = 0.0

        if dollar_volume[i] < config.min_dollar_volume:
            wanted_weight = 0.0

        current_exposure = position * bar_close
        next_order = _build_next_order(
            bar_high=bar_high,
            bar_low=bar_low,
            current_exposure=current_exposure,
            wanted_weight=wanted_weight,
            bar_atr=bar_atr,
            config=config,
        )
        if next_order is not None:
            side, order_price, order_size = next_order
            order_count += 1
            if side == "BUY":
                buy_price = order_price
                buy_size = order_size
                buy_order_count += 1
            else:
                sell_price = order_price
                sell_size = order_size
                sell_order_count += 1

        positions[i] = current_exposure
        equities[i] = cash + current_exposure

    return OneBarLimitResult(
        label=market.label,
        positions=positions,
        equities=equities,
        order_count=order_count,
        fill_count=fill_count,
        buy_order_count=buy_order_count,
        sell_order_count=sell_order_count,
        buy_fill_count=buy_fill_count,
        sell_fill_count=sell_fill_count,
        final_position=position,
        final_cash=cash,
        fills=tuple(fills),
    )


@dataclass(frozen=True, slots=True)
class BacktestCausalBatchRunner:
    """
    Adapter from causal feature frames to the constrained fast-bar executor.
    """

    config: OneBarLimitConfig
    record_fills: bool = True

    def run_markets(
        self, markets: Sequence[OneBarLimitMarketData]
    ) -> tuple[OneBarLimitResult, ...]:
        return tuple(
            simulate_one_bar_limit_market(
                market,
                self.config,
                record_fills=self.record_fills,
            )
            for market in markets
        )

    def run_frames(self, frames: Mapping[str, CausalFrame]) -> tuple[OneBarLimitResult, ...]:
        markets = [self.market_from_frame(label, frame) for label, frame in frames.items()]
        return self.run_markets(markets)

    def run_model(
        self,
        model: CausalFeaturePipeline,
        bars_by_instrument: Mapping[Any, Any],
    ) -> tuple[OneBarLimitResult, ...]:
        if self.config.require_live_compatible and not model.is_live_compatible():
            raise CausalityError("BacktestCausalBatchRunner requires a live-compatible model")
        return self.run_frames(
            {str(label): frame for label, frame in model.compute_batch(bars_by_instrument).items()},
        )

    @staticmethod
    def market_from_frame(label: str, frame: CausalFrame) -> OneBarLimitMarketData:
        high, low, close, target_weight, atr = frame.require(
            "high",
            "low",
            "close",
            "target_weight",
            "atr",
        )
        return OneBarLimitMarketData(
            label=label,
            high=high,
            low=low,
            close=close,
            target_weight=target_weight,
            atr=atr,
            dollar_volume=frame.columns.get("dollar_volume"),
        )


@dataclass(slots=True)
class LiveCausalRunner:
    """
    Online adapter that calls a causal model only after completed bar batches.
    """

    model: CausalFeaturePipeline
    expected_instruments: Sequence[Any] | frozenset[Any] | None = None
    _pending: dict[int, dict[Any, Any]] = field(init=False, default_factory=dict)
    _completed_batches: int = field(init=False, default=0)
    _warmup_bars: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.model.is_live_compatible():
            raise CausalityError("LiveCausalRunner requires a live-compatible model")
        warmup_bars = self.model.warmup_bars()
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be non-negative")
        self._warmup_bars = warmup_bars
        if self.expected_instruments is not None:
            self.expected_instruments = frozenset(self.expected_instruments)

    def on_completed_bar(
        self,
        timestamp: int,
        instrument_id: Any,
        bar: Any,
    ) -> tuple[TargetExposure, ...]:
        """
        Buffer one completed bar until the configured cross-section is complete.
        """
        if self.expected_instruments is None:
            return self.on_completed_bar_batch(timestamp, {instrument_id: bar})

        expected = self.expected_instruments
        if instrument_id not in expected:
            raise ValueError(f"Unexpected live bar instrument {instrument_id!r}")

        timestamp = int(timestamp)
        if any(pending_timestamp < timestamp for pending_timestamp in self._pending):
            raise CausalityError(
                "Received a newer live bar before completing the prior cross-section",
            )

        bucket = self._pending.setdefault(timestamp, {})
        bucket[instrument_id] = bar
        if set(bucket) != expected:
            return ()

        del self._pending[timestamp]
        return self.on_completed_bar_batch(timestamp, bucket)

    def on_completed_bar_batch(
        self,
        timestamp: int,
        bars: Mapping[Any, Any],
    ) -> tuple[TargetExposure, ...]:
        """
        Run the model on bars that are already complete at ``timestamp``.
        """
        timestamp = int(timestamp)
        target_map = self.model.update_bar_batch(timestamp, bars)
        self._completed_batches += 1
        if self._completed_batches < self._warmup_bars:
            return ()
        targets = tuple(target_map.values())
        _validate_live_targets(timestamp, targets)
        return targets


def causal_shift(
    array: CausalArray,
    periods: int,
    *,
    fill_value: float = np.nan,
    name: str | None = None,
) -> CausalArray:
    """
    Return a past-looking shifted feature. Negative shifts are rejected.
    """
    if periods < 0:
        raise CausalityError("Negative shifts require future source rows")

    values = np.full(len(array.values), fill_value, dtype=np.float64)
    available_at = np.array(array.value_time, dtype=np.int64, copy=True)
    if periods == 0:
        values = array.values
        available_at = array.available_at
    elif periods < len(array.values):
        values[periods:] = array.values[:-periods]
        available_at[periods:] = array.available_at[:-periods]

    shifted = CausalArray(
        values=values,
        value_time=array.value_time,
        available_at=available_at,
        name=name or f"{array._label()}_shift_{periods}",
        lineage=(*array.lineage, f"shift({periods})"),
        live_compatible=array.live_compatible,
    )
    shifted.validate_available_at_or_before(shifted.value_time, name=shifted._label())
    return shifted


def causal_trailing_mean(
    array: CausalArray,
    window: int,
    *,
    min_periods: int | None = None,
    centered: bool = False,
    name: str | None = None,
) -> CausalArray:
    """
    Compute an O(n) trailing rolling mean with propagated availability timestamps.
    """
    _validate_trailing_window(window, min_periods, centered)
    min_count = window if min_periods is None else min_periods
    finite = np.isfinite(array.values)
    safe_values = np.where(finite, array.values, 0.0)
    prefix_sum = np.concatenate(([0.0], np.cumsum(safe_values)))
    prefix_count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    values = np.full(len(array.values), np.nan, dtype=np.float64)

    for index in range(len(values)):
        start = max(0, index - window + 1)
        count = int(prefix_count[index + 1] - prefix_count[start])
        if count >= min_count:
            values[index] = float(prefix_sum[index + 1] - prefix_sum[start]) / count

    return _trailing_output(array, values, window, name or f"{array._label()}_mean_{window}")


def causal_trailing_std(
    array: CausalArray,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    centered: bool = False,
    name: str | None = None,
) -> CausalArray:
    """
    Compute an O(n) trailing rolling standard deviation.
    """
    _validate_trailing_window(window, min_periods, centered)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    min_count = window if min_periods is None else min_periods
    finite = np.isfinite(array.values)
    safe_values = np.where(finite, array.values, 0.0)
    prefix_sum = np.concatenate(([0.0], np.cumsum(safe_values)))
    prefix_square_sum = np.concatenate(([0.0], np.cumsum(safe_values * safe_values)))
    prefix_count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    values = np.full(len(array.values), np.nan, dtype=np.float64)

    for index in range(len(values)):
        start = max(0, index - window + 1)
        count = int(prefix_count[index + 1] - prefix_count[start])
        if count >= min_count and count > ddof:
            total = float(prefix_sum[index + 1] - prefix_sum[start])
            square_total = float(prefix_square_sum[index + 1] - prefix_square_sum[start])
            mean = total / count
            variance = max((square_total - count * mean * mean) / (count - ddof), 0.0)
            values[index] = float(np.sqrt(variance))

    return _trailing_output(array, values, window, name or f"{array._label()}_std_{window}")


def causal_same_time_transform(
    inputs: Mapping[str, CausalArray],
    values: FloatArray,
    *,
    name: str,
    live_compatible: bool = True,
) -> CausalArray:
    """
    Wrap a same-timestamp cross-sectional transform and propagate availability.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")

    first_name, first = next(iter(inputs.items()))
    for input_name, array in inputs.items():
        if len(array.values) != len(first.values):
            raise ValueError("All same-time transform inputs must have the same length")
        if not np.array_equal(array.value_time, first.value_time):
            raise CausalityError(
                f"{input_name} is not aligned to the same timestamps as {first_name}",
            )

    output = CausalArray(
        values=np.asarray(values, dtype=np.float64),
        value_time=first.value_time,
        available_at=np.maximum.reduce([array.available_at for array in inputs.values()]),
        name=name,
        lineage=tuple(inputs),
        live_compatible=live_compatible and all(array.live_compatible for array in inputs.values()),
    )
    output.validate_available_at_or_before(output.value_time, name=name)
    return output


def _as_float64(values: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _as_causal_array(values: ArrayInput, name: str, config: OneBarLimitConfig) -> CausalArray:
    if isinstance(values, CausalArray):
        if values.name:
            return values
        return CausalArray(
            values=values.values,
            value_time=values.value_time,
            available_at=values.available_at,
            name=name,
            lineage=values.lineage,
            live_compatible=values.live_compatible,
        )

    if not config.unsafe_assume_causal:
        raise CausalityError(
            f"{name} must be a CausalArray. Raw arrays are accepted only when "
            "OneBarLimitConfig.unsafe_assume_causal=True.",
        )

    array = _as_float64(values, name)
    row_time = np.arange(len(array), dtype=np.int64)
    return CausalArray(
        values=array,
        value_time=row_time,
        available_at=row_time,
        name=name,
    )


def _check_equal_length(*arrays: FloatArray) -> None:
    if not arrays:
        return

    expected = len(arrays[0])
    for array in arrays[1:]:
        if len(array) != expected:
            raise ValueError("All market arrays must have the same length")


def _validate_market_causality(
    label: str,
    decision_time: IntArray,
    columns: Sequence[tuple[str, CausalArray]],
    config: OneBarLimitConfig,
) -> None:
    for name, array in columns:
        if not np.array_equal(array.value_time, decision_time):
            raise CausalityError(
                f"{label}.{name} value_time must match close decision timestamps",
            )
        if config.require_live_compatible and not array.live_compatible:
            raise CausalityError(f"{label}.{name} is not declared live-compatible")
        array.validate_available_at_or_before(decision_time, name=f"{label}.{name}")


def _validate_live_targets(timestamp: int, targets: Sequence[TargetExposure]) -> None:
    for target in targets:
        if target.available_at > target.decision_time:
            raise CausalityError(
                f"{target.instrument_id} target is available after its decision timestamp: "
                f"available_at={target.available_at}, decision_time={target.decision_time}",
            )
        if target.decision_time > timestamp:
            raise CausalityError(
                f"{target.instrument_id} target decision_time={target.decision_time} "
                f"is after completed live bar timestamp={timestamp}",
            )


def _validate_trailing_window(
    window: int,
    min_periods: int | None,
    centered: bool,
) -> None:
    if centered:
        raise CausalityError("Centered rolling windows require future source rows")
    if window <= 0:
        raise ValueError("window must be positive")
    if min_periods is not None and (min_periods <= 0 or min_periods > window):
        raise ValueError("min_periods must be in the range [1, window]")


def _trailing_output(
    array: CausalArray,
    values: FloatArray,
    window: int,
    name: str,
) -> CausalArray:
    output = CausalArray(
        values=values,
        value_time=array.value_time,
        available_at=_rolling_max_int(array.available_at, window),
        name=name,
        lineage=(*array.lineage, f"trailing({window})"),
        live_compatible=array.live_compatible,
    )
    output.validate_available_at_or_before(output.value_time, name=name)
    return output


def _rolling_max_int(values: IntArray, window: int) -> IntArray:
    # A small monotonic deque keeps availability propagation O(n), even for long windows.
    result = np.empty(len(values), dtype=np.int64)
    indices: deque[int] = deque()
    for index, value in enumerate(values):
        while indices and indices[0] <= index - window:
            indices.popleft()
        while indices and values[indices[-1]] <= value:
            indices.pop()
        indices.append(index)
        result[index] = values[indices[0]]
    return result


def _buy_crossed(bar_low: float, limit_price: float, strict: bool) -> bool:
    return bar_low < limit_price if strict else bar_low <= limit_price


def _sell_crossed(bar_high: float, limit_price: float, strict: bool) -> bool:
    return bar_high > limit_price if strict else bar_high >= limit_price


def _build_next_order(
    *,
    bar_high: float,
    bar_low: float,
    current_exposure: float,
    wanted_weight: float,
    bar_atr: float,
    config: OneBarLimitConfig,
) -> tuple[str, float, float] | None:
    if not np.isfinite(wanted_weight) or not np.isfinite(bar_atr) or bar_atr <= 0.0:
        return None

    target_exposure = wanted_weight * config.bet_size
    exposure_diff = target_exposure - current_exposure
    if exposure_diff > 0.0:
        raw_price = bar_low * (1.0 - bar_atr * config.limit_entry_atr_multiple)
        raw_size = exposure_diff / raw_price if raw_price > 0.0 else 0.0
        return _build_order("BUY", raw_price, raw_size, config)

    if exposure_diff < 0.0:
        raw_price = bar_high * (1.0 + bar_atr * config.limit_entry_atr_multiple)
        raw_size = -exposure_diff / raw_price if raw_price > 0.0 else 0.0
        return _build_order("SELL", raw_price, raw_size, config)

    return None


def _build_order(
    side: str,
    raw_price: float,
    raw_size: float,
    config: OneBarLimitConfig,
) -> tuple[str, float, float] | None:
    order_price = _round_optional(raw_price, config.price_precision)
    order_size = _round_optional(raw_size, config.quantity_precision)
    if not _is_valid_order(order_price, order_size, config.min_quantity):
        return None
    return side, order_price, order_size


def _round_optional(value: float, precision: int | None) -> float:
    if precision is None:
        return value
    return round(value, precision)


def _is_valid_order(price: float, quantity: float, min_quantity: float) -> bool:
    return (
        price > 0.0
        and quantity > 0.0
        and quantity >= min_quantity
        and np.isfinite(price)
        and np.isfinite(quantity)
    )
