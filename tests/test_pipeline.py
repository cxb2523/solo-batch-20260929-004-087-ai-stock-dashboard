"""Fixed-fixture assertions for the fetch -> indicators -> ML -> backtest chain."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

import pipeline as pl


@pytest.fixture
def fixed_ohlcv():
    """80 deterministic business days of OHLCV, strictly increasing close."""
    n = 80
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(
        np.round(100.0 + np.arange(n) * 0.5 + np.sin(np.arange(n) / 5.0), 4),
        index=idx,
    )
    open_ = (close.shift(1).fillna(close.iloc[0]) + 0.1).round(4)
    high = (np.maximum(open_, close) + 0.3).round(4)
    low = (np.minimum(open_, close) - 0.3).round(4)
    volume = pd.Series(1_000 + np.arange(n) * 7, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


@pytest.fixture
def indicators(fixed_ohlcv):
    return pl.calculate_technical_indicators(fixed_ohlcv)


def test_frame_sorted_by_date_before_indicators(fixed_ohlcv):
    shuffled = fixed_ohlcv.sample(frac=1.0, random_state=0)
    out = pl.calculate_technical_indicators(shuffled)
    assert out.index.is_monotonic_increasing
    assert list(out.index) == list(fixed_ohlcv.index)


def test_insufficient_windows_are_nan_not_zero(indicators):
    assert indicators["SMA_20"].iloc[:19].isna().all()
    assert indicators["SMA_50"].iloc[:49].isna().all()
    assert indicators["SMA_200"].isna().all()
    assert (indicators[["SMA_20", "SMA_50"]].iloc[:19] == 0).sum().sum() == 0
    assert indicators["RSI"].iloc[:14].isna().all()
    assert indicators["MACD"].iloc[:25].isna().all()
    assert indicators["MACD_signal"].iloc[:33].isna().all()


def test_sma20_exact_values(indicators, fixed_ohlcv):
    expected_19 = fixed_ohlcv["Close"].iloc[:20].mean()
    assert indicators["SMA_20"].iloc[19] == pytest.approx(expected_19)
    i = 50
    expected_50 = fixed_ohlcv["Close"].iloc[i - 19 : i + 1].mean()
    assert indicators["SMA_20"].iloc[i] == pytest.approx(expected_50)
    assert not np.isnan(indicators["SMA_20"].iloc[i])


def test_rsi_values(indicators, fixed_ohlcv):
    close = fixed_ohlcv["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    expected_rsi = 100 - 100 / (1 + gain / loss)
    pd.testing.assert_series_equal(
        indicators["RSI"], expected_rsi, check_names=False
    )
    assert (indicators["RSI"].dropna() == 100.0).all()


def test_macd_values(indicators, fixed_ohlcv):
    close = fixed_ohlcv["Close"]
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    pd.testing.assert_series_equal(indicators["MACD"], macd, check_names=False)
    pd.testing.assert_series_equal(
        indicators["MACD_signal"], signal, check_names=False
    )
    pd.testing.assert_series_equal(
        indicators["MACD_histogram"], macd - signal, check_names=False
    )


def test_bollinger_and_atr(indicators, fixed_ohlcv):
    window = fixed_ohlcv["Close"].iloc[:20]
    assert indicators["BB_middle"].iloc[19] == pytest.approx(window.mean())
    assert indicators["BB_upper"].iloc[19] == pytest.approx(
        window.mean() + 2 * window.std()
    )
    assert indicators["BB_lower"].iloc[19] == pytest.approx(
        window.mean() - 2 * window.std()
    )
    assert indicators["ATR"].iloc[:13].isna().all()
    assert indicators["ATR"].iloc[13] > 0


def test_rsi_flat_market_is_fifty():
    n = 30
    idx = pd.bdate_range("2024-01-01", periods=n)
    flat = pd.DataFrame(
        {
            "Open": np.full(n, 50.0),
            "High": np.full(n, 50.0),
            "Low": np.full(n, 50.0),
            "Close": np.full(n, 50.0),
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )
    out = pl.calculate_technical_indicators(flat)
    assert (out["RSI"].dropna() == 50.0).all()


def test_label_is_next_close_and_only_tail_dropped(indicators):
    label = pl.make_prediction_label(indicators)
    assert label.iloc[0] == pytest.approx(indicators["Close"].iloc[1])
    assert label.iloc[40] == pytest.approx(indicators["Close"].iloc[41])
    assert pd.isna(label.iloc[-1])


def test_prediction_dataset_alignment_no_lookahead(indicators):
    x, y = pl.build_prediction_dataset(indicators)
    assert x.index.equals(y.index)
    assert not x.isna().any().any()
    assert not y.isna().any()
    assert indicators.index[-1] not in y.index
    for ts in y.index:
        expected = indicators["Close"].loc[ts:].iloc[1]
        assert y.loc[ts] == pytest.approx(expected)
    first_pos = indicators.index.get_loc(x.index[0])
    assert x["Close_lag_1"].iloc[0] == pytest.approx(
        indicators["Close"].iloc[first_pos - 1]
    )


def test_backtest_strict_dropna_first_last_alignment(fixed_ohlcv):
    out = pl.calculate_technical_indicators(fixed_ohlcv)
    signal = pd.Series(np.nan, index=out.index)
    signal.loc[out["SMA_20"].notna()] = 1.0
    bt = pl.backtest_returns(out, signal)

    assert not bt.isna().any().any()
    assert bt.index.is_monotonic_increasing
    assert bt.index[0] == out.index[19]
    assert bt.index[-1] == out.index[-2]

    closes = out["Close"]
    first = bt.index[0]
    expected_first = closes.loc[first:].iloc[1] / closes.loc[first] - 1
    assert bt["Forward_Return"].iloc[0] == pytest.approx(expected_first)
    last = bt.index[-1]
    expected_last = closes.iloc[-1] / closes.loc[last] - 1
    assert bt["Forward_Return"].iloc[-1] == pytest.approx(expected_last)
    assert bt["Strategy_Return"].iloc[0] == pytest.approx(expected_first)
    assert list(bt.columns) == ["Forward_Return", "Signal", "Strategy_Return"]


def test_lru_cache_keyed_by_triplet_and_lru_eviction():
    cache = pl.LRUQuoteCache(maxsize=2)

    def snap(ticker, start, end, adjust):
        return pl.QuoteSnapshot(
            ticker=ticker,
            start=start,
            end=end,
            adjust=adjust,
            data=pd.DataFrame({"Close": [1.0]}),
            info={},
            fetched_at=0.0,
        )

    k1 = cache.make_key("aapl", "2024-01-01", "2024-02-01", "qfq")
    k2 = cache.make_key("AAPL", "2024-01-01", "2024-03-01", "qfq")
    k3 = cache.make_key("msft", "2024-01-01", "2024-02-01", "qfq")
    k4 = cache.make_key("AAPL", "2024-01-01", "2024-02-01", "hfq")
    k1_norm = cache.make_key("AAPL", "2024-01-01", "2024-02-01", "qfq")
    assert k1 == k1_norm
    assert len({k1, k2, k3, k4}) == 4

    cache.put(k1, snap("AAPL", "a", "b", "qfq"))
    cache.put(k2, snap("AAPL", "a", "c", "qfq"))
    assert cache.get(k1) is not None
    cache.put(k3, snap("MSFT", "a", "b", "qfq"))
    assert k2 not in cache
    assert k1 in cache and k3 in cache


def _make_ohlcv(n=10):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": np.full(n, 10.0),
            "High": np.full(n, 11.0),
            "Low": np.full(n, 9.0),
            "Close": np.linspace(10, 11, n),
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_fetcher_retries_three_times_then_serves_stale(monkeypatch):
    fetcher = pl.QuoteFetcher()
    calls = {"n": 0}

    def good(ticker, start, end, adjust):
        return _make_ohlcv()

    def broken(ticker, start, end, adjust):
        calls["n"] += 1
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(fetcher, "_fetch_info", lambda ticker: {})

    monkeypatch.setattr(fetcher, "_download", good)
    fresh = fetcher.fetch("AAPL", "2024-01-01", "2024-02-01", "qfq")
    assert fresh.stale is False
    assert len(fresh.data) == 10

    monkeypatch.setattr(fetcher, "_download", broken)
    stale = fetcher.fetch(
        "AAPL", "2024-01-01", "2024-02-01", "qfq", force_refresh=True
    )
    assert calls["n"] == pl.FETCH_ATTEMPTS
    assert stale.stale is True
    assert "simulated timeout" in (stale.error or "")
    assert stale.fetched_at == fresh.fetched_at


def test_fetcher_raises_when_no_snapshot(monkeypatch):
    fetcher = pl.QuoteFetcher()

    def broken(ticker, start, end, adjust):
        raise TimeoutError("no network")

    monkeypatch.setattr(fetcher, "_download", broken)
    monkeypatch.setattr(fetcher, "_fetch_info", lambda ticker: {})
    with pytest.raises(RuntimeError, match="failed to fetch"):
        fetcher.fetch("NOPE", "2024-01-01", "2024-02-01", "qfq")


def test_fetcher_single_flight_same_key(monkeypatch):
    fetcher = pl.QuoteFetcher()
    downloads = {"n": 0}

    def slow(ticker, start, end, adjust):
        downloads["n"] += 1
        time.sleep(0.2)
        return _make_ohlcv()

    monkeypatch.setattr(fetcher, "_download", slow)
    monkeypatch.setattr(fetcher, "_fetch_info", lambda ticker: {})

    results = []

    def worker():
        results.append(fetcher.fetch("AAPL", "2024-01-01", "2024-02-01", "qfq"))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)
    assert downloads["n"] == 1
    assert len(results) == 2
    assert all(r.stale is False for r in results)
