"""Fixed-fixture assertions for the fetch -> indicator -> label/backtest chain.

Run with:  python -m pytest -q
"""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from pipeline import (
    LRUCache,
    DataFetcher,
    align_features_target,
    calculate_technical_indicators,
    forward_returns,
    make_label_target,
    normalize_key,
    prepare_ml_features,
    run_backtest,
)


@pytest.fixture
def ohlcv():
    """60 business days. Close rises 1.0 per day (100 -> 159), deterministic
    intraday structure, deliberately shuffled on the date axis."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(np.arange(100.0, 160.0), index=dates)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(60, 10_000.0),
        },
        index=dates,
    )
    return df.iloc[::-1]  # descending dates on purpose


@pytest.fixture
def indicators(ohlcv):
    return calculate_technical_indicators(ohlcv)


def test_indicators_sorted_by_date_first(ohlcv, indicators):
    assert not ohlcv.index.is_monotonic_increasing
    assert indicators.index.is_monotonic_increasing
    assert list(indicators["Close"]) == list(np.arange(100.0, 160.0))


def test_rolling_means_are_nan_until_window_full(indicators):
    assert indicators["SMA_20"].iloc[:19].isna().all()
    assert indicators["SMA_20"].iloc[19] == pytest.approx(np.arange(100.0, 120.0).mean())

    assert indicators["SMA_50"].iloc[:49].isna().all()
    assert indicators["SMA_50"].iloc[49] == pytest.approx(np.arange(100.0, 150.0).mean())

    # 60 bars cannot fill a 200 window -> whole column NaN, never zero.
    assert indicators["SMA_200"].isna().all()

    assert indicators["Volume_SMA"].iloc[:19].isna().all()
    assert indicators["Volume_SMA"].iloc[19] == pytest.approx(10_000.0)
    assert not (indicators["Volume_SMA"].dropna() == 0).any()


def test_rsi_values_on_fixture(indicators):
    # First 14 rows empty (window insufficient), not zero.
    assert indicators["RSI"].iloc[:14].isna().all()
    # Close rises exactly 1/day -> no losses -> RSI 100.
    assert indicators["RSI"].iloc[14:].eq(100.0).all()


def test_rsi_handles_losses(ohlcv):
    down = ohlcv.sort_index().copy()
    down["Close"] = np.arange(160.0, 100.0, -1.0)
    ind = calculate_technical_indicators(down)
    assert ind["RSI"].iloc[:14].isna().all()
    assert ind["RSI"].iloc[14:].eq(0.0).all()


def test_macd_warmup_is_nan_not_zero(indicators):
    assert indicators["EMA_26"].iloc[:25].isna().all()
    assert indicators["MACD"].iloc[:25].isna().all()
    # MACD starts at bar 25; the signal needs 9 MACD values -> bars 25..32
    # are NaN and the first signal value appears at bar 33.
    assert indicators["MACD_signal"].iloc[:33].isna().all()
    assert pd.notna(indicators["MACD_signal"].iloc[33])
    assert pd.notna(indicators["MACD_histogram"].iloc[33])


def test_bollinger_and_atr_warmup(indicators):
    assert indicators["BB_upper"].iloc[:19].isna().all()
    assert indicators["BB_lower"].iloc[:19].isna().all()
    assert indicators["ATR"].iloc[:13].isna().all()
    # H-L is always 2.0 and close-open gaps are within it -> TR = 2.
    assert indicators["ATR"].iloc[13] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Label shift discipline (no look-ahead; label side drops only the tail)
# ---------------------------------------------------------------------------

def test_label_is_forward_shift_and_nan_only_at_tail(indicators):
    target = make_label_target(indicators["Close"])
    assert target.iloc[:-1].tolist() == indicators["Close"].iloc[1:].tolist()
    assert pd.isna(target.iloc[-1])
    assert target.iloc[:-1].notna().all()

    fwd = forward_returns(indicators["Close"])
    assert fwd.iloc[0] == pytest.approx(1.0 / 100.0)
    assert pd.isna(fwd.iloc[-1])


def test_align_features_target_drops_only_tail(ohlcv):
    feats = prepare_ml_features(ohlcv)
    target = make_label_target(feats["Close"])
    X, y, tail_dropped = align_features_target(
        feats[["SMA_20", "SMA_50"]], target
    )

    # SMA_50 warm-up drops 49 head rows on the feature side;
    # exactly 1 tail row is dropped on the label side.
    assert tail_dropped == 1
    assert len(X) == 60 - 49 - 1
    assert X.index.equals(y.index)
    assert X.index[0] == feats.index[49]
    assert X.index[-1] == feats.index[-2]
    assert y.iloc[0] == pytest.approx(feats["Close"].iloc[50])
    assert X.notna().all().all() and y.notna().all()


# ---------------------------------------------------------------------------
# Backtest alignment
# ---------------------------------------------------------------------------

def test_backtest_strict_dropna_and_first_last_alignment(indicators):
    position = pd.Series(1.0, index=indicators.index)
    bt = run_backtest(indicators["Close"], position)

    assert bt.notna().all().all()
    assert len(bt) == 59
    assert bt.index[0] == indicators.index[0]
    assert bt.index[-1] == indicators.index[-2]

    expected = indicators["Close"].shift(-1) / indicators["Close"] - 1
    pd.testing.assert_series_equal(
        bt["forward_return"], expected.iloc[:-1], check_names=False
    )
    np.testing.assert_allclose(
        bt["strategy_return"].to_numpy(), bt["forward_return"].to_numpy()
    )
    assert bt["equity"].iloc[-1] == pytest.approx(159.0 / 100.0)
    assert bt["benchmark"].iloc[-1] == pytest.approx(159.0 / 100.0)


def test_backtest_flat_position_is_zero_return(indicators):
    position = pd.Series(0.0, index=indicators.index)
    bt = run_backtest(indicators["Close"], position)
    assert len(bt) == 59
    assert bt["strategy_return"].eq(0).all()
    assert bt["equity"].eq(1).all()


def test_backtest_with_nan_position_strictly_removed(indicators):
    position = pd.Series(np.nan, index=indicators.index)
    position.iloc[10:20] = 1.0
    bt = run_backtest(indicators["Close"], position)
    assert len(bt) == 10
    assert bt.index[0] == indicators.index[10]
    assert bt.index[-1] == indicators.index[19]


# ---------------------------------------------------------------------------
# LRU + fetch layer
# ---------------------------------------------------------------------------

def _frame(n=30, start="2024-01-01", price=100.0):
    idx = pd.bdate_range(start, periods=n)
    close = price + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_lru_cache_eviction_and_isolation():
    cache = LRUCache(maxsize=2)
    ka = ("A", "2024-01-01", "2024-02-01", "raw")
    kb = ("B", "2024-01-01", "2024-02-01", "raw")
    kc = ("A", "2024-03-01", "2024-04-01", "all")
    cache.set(ka, 1)
    cache.set(kb, 2)
    assert cache.get(ka) == 1  # touch A so B becomes LRU
    cache.set(kc, 3)
    assert len(cache) == 2
    assert cache.get(kb) is None
    assert cache.get(ka) == 1
    assert cache.get(kc) == 3


def test_cache_key_contains_all_three_dimensions():
    base = normalize_key("aapl", "2024-01-01", "2024-02-01", "raw")
    assert base == ("AAPL", "2024-01-01", "2024-02-01", "raw")
    assert normalize_key("AAPL", "2024-01-01", "2024-02-01", "all") != base
    assert normalize_key("AAPL", "2024-01-01", "2024-03-01", "raw") != base
    assert normalize_key("MSFT", "2024-01-01", "2024-02-01", "raw") != base


def test_fetcher_calls_raw_once_per_triple():
    calls = []

    def raw(ticker, start, end, adjust):
        calls.append(normalize_key(ticker, start, end, adjust))
        return _frame(), {"symbol": ticker}

    fetcher = DataFetcher(raw_fetcher=raw, retry_sleep=0)
    r1 = fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "raw")
    r2 = fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "raw")
    r3 = fetcher.fetch("MSFT", "2024-01-01", "2024-02-15", "raw")

    assert len(calls) == 2
    assert r1.cached is False
    assert r2.cached is True
    assert r3.cached is False
    assert r1.data.index.is_monotonic_increasing


def test_fetcher_no_cross_ticker_range_adjust_reuse():
    served = {}

    def raw(ticker, start, end, adjust):
        key = normalize_key(ticker, start, end, adjust)
        served[key] = served.get(key, 0) + 1
        return _frame(price=(hash(key) % 50) + 10.0), {}

    fetcher = DataFetcher(raw_fetcher=raw, retry_sleep=0)
    a = fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "raw")
    b = fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "all")
    c = fetcher.fetch("AAPL", "2024-03-01", "2024-04-01", "raw")
    d = fetcher.fetch("MSFT", "2024-01-01", "2024-02-15", "raw")

    assert len(fetcher.cache) == 4
    assert len({a.key, b.key, c.key, d.key}) == 4
    for _ in range(3):
        fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "raw")
        fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "all")
        fetcher.fetch("AAPL", "2024-03-01", "2024-04-01", "raw")
        fetcher.fetch("MSFT", "2024-01-01", "2024-02-15", "raw")
    assert all(v == 1 for v in served.values())


def test_three_timeouts_degrade_to_last_snapshot():
    state = {"fail": False, "calls": 0}

    def raw(ticker, start, end, adjust):
        state["calls"] += 1
        if state["fail"]:
            raise TimeoutError("simulated quote timeout")
        return _frame(), {}

    fetcher = DataFetcher(raw_fetcher=raw, retry_sleep=0)
    good = fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "raw")
    assert good.ok and not good.stale

    # A different triple fails three times -> fall back to ticker snapshot.
    state["fail"] = True
    stale = fetcher.fetch("AAPL", "2024-03-01", "2024-04-01", "raw")
    assert stale.ok and stale.stale
    assert stale.attempts == 3
    assert state["calls"] == 1 + 3
    pd.testing.assert_frame_equal(stale.data, good.data)
    assert "snapshot" in stale.freshness_text()


def test_all_attempts_fail_without_snapshot_returns_error():
    def raw(ticker, start, end, adjust):
        raise TimeoutError("boom")

    fetcher = DataFetcher(raw_fetcher=raw, retry_sleep=0)
    result = fetcher.fetch("NOPE", "2024-01-01", "2024-02-15", "raw")
    assert not result.ok
    assert result.attempts == 3
    assert result.freshness_text() is None


def test_concurrent_requests_coalesced_per_triple():
    calls = []
    lock = threading.Lock()

    def raw(ticker, start, end, adjust):
        time.sleep(0.1)
        with lock:
            calls.append(normalize_key(ticker, start, end, adjust))
        return _frame(), {}

    fetcher = DataFetcher(raw_fetcher=raw, retry_sleep=0)
    results = []

    def worker():
        results.append(fetcher.fetch("AAPL", "2024-01-01", "2024-02-15", "raw"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert sum(r.cached for r in results) == 3
    assert all(r.ok for r in results)
