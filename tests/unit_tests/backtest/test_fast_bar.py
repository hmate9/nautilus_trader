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

import numpy as np
import pytest

from nautilus_trader.backtest.fast_bar import BacktestCausalBatchRunner
from nautilus_trader.backtest.fast_bar import CausalArray
from nautilus_trader.backtest.fast_bar import CausalFrame
from nautilus_trader.backtest.fast_bar import CausalityError
from nautilus_trader.backtest.fast_bar import LiveCausalRunner
from nautilus_trader.backtest.fast_bar import OneBarLimitConfig
from nautilus_trader.backtest.fast_bar import OneBarLimitMarketData
from nautilus_trader.backtest.fast_bar import TargetExposure
from nautilus_trader.backtest.fast_bar import causal_shift
from nautilus_trader.backtest.fast_bar import causal_trailing_mean
from nautilus_trader.backtest.fast_bar import simulate_one_bar_limit_market


def causal(
    values: list[float],
    *,
    name: str,
    value_time: list[int] | None = None,
    available_at: list[int] | None = None,
    live_compatible: bool = True,
) -> CausalArray:
    timestamps = np.arange(len(values), dtype=np.int64) if value_time is None else value_time
    return CausalArray(
        values=np.array(values, dtype=np.float64),
        value_time=np.asarray(timestamps, dtype=np.int64),
        available_at=None if available_at is None else np.asarray(available_at, dtype=np.int64),
        name=name,
        live_compatible=live_compatible,
    )


def market_from_values(
    *,
    high: list[float],
    low: list[float],
    close: list[float],
    target_weight: list[float],
    atr: list[float],
    dollar_volume: list[float] | None = None,
) -> OneBarLimitMarketData:
    return OneBarLimitMarketData(
        label="TEST",
        high=causal(high, name="high"),
        low=causal(low, name="low"),
        close=causal(close, name="close"),
        target_weight=causal(target_weight, name="target_weight"),
        atr=causal(atr, name="atr"),
        dollar_volume=None
        if dollar_volume is None
        else causal(dollar_volume, name="dollar_volume"),
    )


def test_simulate_one_bar_limit_market_requires_strict_crossing_by_default():
    market = market_from_values(
        high=[10.0, 10.0],
        low=[10.0, 9.0],
        close=[10.0, 10.0],
        target_weight=[1.0, 0.0],
        atr=[0.4, 0.4],
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    result = simulate_one_bar_limit_market(market, config)

    assert result.order_count == 1
    assert result.fill_count == 0
    assert result.final_position == 0.0
    np.testing.assert_array_equal(result.equities, np.array([0.0, 0.0]))


def test_simulate_one_bar_limit_market_can_use_non_strict_crossing():
    market = market_from_values(
        high=[10.0, 10.0],
        low=[10.0, 9.0],
        close=[10.0, 10.0],
        target_weight=[1.0, 0.0],
        atr=[0.4, 0.4],
    )
    config = OneBarLimitConfig(
        bet_size=100.0,
        limit_entry_atr_multiple=0.25,
        strict_crossing=False,
    )

    result = simulate_one_bar_limit_market(market, config)

    assert result.fill_count == 1
    assert result.buy_fill_count == 1
    assert result.fills[0].side == "BUY"
    assert result.fills[0].price == 9.0


def test_simulate_one_bar_limit_market_fills_prior_order_then_places_next_order():
    market = market_from_values(
        high=[10.0, 10.0, 12.0],
        low=[10.0, 8.0, 10.0],
        close=[10.0, 10.0, 10.0],
        target_weight=[1.0, 0.0, 0.0],
        atr=[0.4, 0.4, 0.4],
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    result = simulate_one_bar_limit_market(market, config)

    assert result.order_count == 3
    assert result.fill_count == 2
    assert result.buy_fill_count == 1
    assert result.sell_fill_count == 1
    np.testing.assert_allclose(result.positions, np.array([0.0, 1000.0 / 9.0, 1000.0 / 99.0]))
    np.testing.assert_allclose(result.equities, np.array([0.0, 100.0 / 9.0, 2100.0 / 99.0]))


def test_simulate_one_bar_limit_market_validates_input_shapes():
    market = OneBarLimitMarketData(
        label="TEST",
        high=causal([10.0], name="high"),
        low=causal([10.0, 9.0], name="low"),
        close=causal([10.0], name="close"),
        target_weight=causal([1.0], name="target_weight"),
        atr=causal([0.4], name="atr"),
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    with pytest.raises(ValueError, match="same length"):
        simulate_one_bar_limit_market(market, config)


def test_simulate_one_bar_limit_market_can_apply_order_precision():
    market = market_from_values(
        high=[10.0, 10.0],
        low=[10.0, 9.0],
        close=[10.0, 10.0],
        target_weight=[0.0000000001, 0.0],
        atr=[0.4, 0.4],
    )
    config = OneBarLimitConfig(
        bet_size=100.0,
        limit_entry_atr_multiple=0.25,
        quantity_precision=8,
        min_quantity=1e-9,
    )

    result = simulate_one_bar_limit_market(market, config)

    assert result.order_count == 0
    assert result.fill_count == 0


def test_simulate_one_bar_limit_market_rejects_raw_arrays_by_default():
    market = OneBarLimitMarketData(
        label="TEST",
        high=np.array([10.0]),
        low=np.array([10.0]),
        close=np.array([10.0]),
        target_weight=np.array([1.0]),
        atr=np.array([0.4]),
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    with pytest.raises(CausalityError, match="must be a CausalArray"):
        simulate_one_bar_limit_market(market, config)


def test_simulate_one_bar_limit_market_can_accept_raw_arrays_with_unsafe_flag():
    market = OneBarLimitMarketData(
        label="TEST",
        high=np.array([10.0, 10.0]),
        low=np.array([10.0, 8.0]),
        close=np.array([10.0, 10.0]),
        target_weight=np.array([1.0, 0.0]),
        atr=np.array([0.4, 0.4]),
    )
    config = OneBarLimitConfig(
        bet_size=100.0,
        limit_entry_atr_multiple=0.25,
        unsafe_assume_causal=True,
    )

    result = simulate_one_bar_limit_market(market, config)

    assert result.fill_count == 1


def test_simulate_one_bar_limit_market_rejects_future_available_features():
    market = OneBarLimitMarketData(
        label="TEST",
        high=causal([10.0, 10.0], name="high", value_time=[10, 20]),
        low=causal([10.0, 8.0], name="low", value_time=[10, 20]),
        close=causal([10.0, 10.0], name="close", value_time=[10, 20]),
        target_weight=causal(
            [1.0, 0.0],
            name="target_weight",
            value_time=[10, 20],
            available_at=[10, 30],
        ),
        atr=causal([0.4, 0.4], name="atr", value_time=[10, 20]),
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    with pytest.raises(CausalityError, match=r"TEST\.target_weight"):
        simulate_one_bar_limit_market(market, config)


def test_simulate_one_bar_limit_market_rejects_misaligned_feature_times():
    market = OneBarLimitMarketData(
        label="TEST",
        high=causal([10.0, 10.0], name="high", value_time=[10, 20]),
        low=causal([10.0, 8.0], name="low", value_time=[10, 20]),
        close=causal([10.0, 10.0], name="close", value_time=[10, 20]),
        target_weight=causal([1.0, 0.0], name="target_weight", value_time=[10, 30]),
        atr=causal([0.4, 0.4], name="atr", value_time=[10, 20]),
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    with pytest.raises(CausalityError, match="value_time"):
        simulate_one_bar_limit_market(market, config)


def test_simulate_one_bar_limit_market_rejects_non_live_compatible_inputs():
    market = market_from_values(
        high=[10.0],
        low=[10.0],
        close=[10.0],
        target_weight=[1.0],
        atr=[0.4],
    )
    market = OneBarLimitMarketData(
        label=market.label,
        high=market.high,
        low=market.low,
        close=market.close,
        target_weight=causal([1.0], name="target_weight", live_compatible=False),
        atr=market.atr,
        dollar_volume=market.dollar_volume,
    )
    config = OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25)

    with pytest.raises(CausalityError, match="not declared live-compatible"):
        simulate_one_bar_limit_market(market, config)


def test_causal_shift_rejects_future_shift():
    feature = causal([1.0, 2.0], name="feature")

    with pytest.raises(CausalityError, match="Negative shifts"):
        causal_shift(feature, -1)


def test_causal_trailing_mean_rejects_centered_window():
    feature = causal([1.0, 2.0, 3.0], name="feature")

    with pytest.raises(CausalityError, match="Centered rolling"):
        causal_trailing_mean(feature, 3, centered=True)


def test_backtest_causal_batch_runner_runs_causal_frames():
    frame = CausalFrame(
        {
            "high": causal([10.0, 10.0], name="high"),
            "low": causal([10.0, 8.0], name="low"),
            "close": causal([10.0, 10.0], name="close"),
            "target_weight": causal([1.0, 0.0], name="target_weight"),
            "atr": causal([0.4, 0.4], name="atr"),
        },
    )
    runner = BacktestCausalBatchRunner(
        OneBarLimitConfig(bet_size=100.0, limit_entry_atr_multiple=0.25),
    )

    (result,) = runner.run_frames({"TEST": frame})

    assert result.fill_count == 1


class _LiveModel:
    def __init__(self, *, available_at_offset: int = 0, warmup: int = 1) -> None:
        self.available_at_offset = available_at_offset
        self.warmup = warmup
        self.calls = 0

    def warmup_bars(self) -> int:
        return self.warmup

    def is_live_compatible(self) -> bool:
        return True

    def compute_batch(self, bars_by_instrument):
        return {}

    def update_bar_batch(self, timestamp, bars):
        self.calls += 1
        return {
            instrument_id: TargetExposure(
                instrument_id=instrument_id,
                target_weight=1.0,
                atr=0.4,
                decision_time=timestamp,
                available_at=timestamp + self.available_at_offset,
            )
            for instrument_id in bars
        }


def test_live_causal_runner_buffers_until_completed_cross_section():
    model = _LiveModel()
    runner = LiveCausalRunner(model=model, expected_instruments=("A", "B"))

    assert runner.on_completed_bar(100, "A", object()) == ()
    targets = runner.on_completed_bar(100, "B", object())

    assert model.calls == 1
    assert {target.instrument_id for target in targets} == {"A", "B"}


def test_live_causal_runner_rejects_future_targets():
    runner = LiveCausalRunner(model=_LiveModel(available_at_offset=1))

    with pytest.raises(CausalityError, match="available after"):
        runner.on_completed_bar_batch(100, {"A": object()})


def test_live_causal_runner_respects_model_warmup():
    model = _LiveModel(warmup=2)
    runner = LiveCausalRunner(model=model)

    assert runner.on_completed_bar_batch(100, {"A": object()}) == ()
    targets = runner.on_completed_bar_batch(200, {"A": object()})

    assert model.calls == 2
    assert len(targets) == 1
