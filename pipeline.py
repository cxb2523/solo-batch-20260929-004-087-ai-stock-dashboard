"""Data pipeline for the stock dashboard.

Layer 1 - fetching (:class:`DataFetcher`): every quote request is keyed by
the ``(ticker, start, end, adjust)`` triple and cached in an explicit,
thread-safe LRU. Repeated requests for the same ticker/range/adjustment
never hit the network twice. After three failed attempts the most recent
successful snapshot is returned and flagged as stale.

Layer 2 - indicators (:func:`calculate_technical_indicators`), labels
(:func:`make_label_target`, :func:`align_features_target`) and backtests
(:func:`run_backtest`): OHLCV rows are always sorted by date first,
rolling/EWM windows that are too short stay ``NaN`` (never zero) and the
forward-looking label/return series share the exact same shift direction.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# yfinance ``actions`` argument mapping for the supported adjustments.
ADJUST_ACTIONS = {
    "raw": None,
    "split": "dividends",
    "dividend": "splits",
    "all": True,
}

ADJUST_LABELS = {
    "raw": "Unadjusted",
    "split": "Split-adjusted",
    "dividend": "Dividend-adjusted",
    "all": "Fully adjusted",
}


def normalize_key(ticker: str, start, end, adjust: str) -> Tuple[str, str, str, str]:
    """Build the canonical cache key.

    Ticker, start date, end date and adjustment mode are all part of the
    key, so two stocks (or two ranges / adjustments) can never share quotes.
    """
    t = str(ticker).strip().upper()
    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = pd.Timestamp(end).strftime("%Y-%m-%d")
    a = str(adjust).strip().lower()
    if a not in ADJUST_ACTIONS:
        raise ValueError(f"Unsupported adjust mode: {adjust!r}")
    return (t, s, e, a)


class LRUCache:
    """Explicit, thread-safe LRU cache."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._store: "OrderedDict[tuple, object]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def set(self, key, value) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def keys(self):
        with self._lock:
            return list(self._store.keys())

    def discard(self, key) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0


@dataclass
class FetchResult:
    """Outcome of a fetch attempt. ``stale`` marks a snapshot served after
    the live request failed; ``cached`` marks an LRU hit."""

    data: Optional[pd.DataFrame]
    info: dict
    key: Tuple[str, str, str, str]
    cached: bool = False
    stale: bool = False
    attempts: int = 0
    fetched_at: Optional[float] = None
    as_of: Optional[pd.Timestamp] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.data is not None and not self.data.empty

    def freshness_text(self) -> Optional[str]:
        if not self.ok:
            return None
        when = self.as_of.strftime("%Y-%m-%d") if self.as_of is not None else "?"
        if self.stale:
            return (
                f"⚠️ Live quote timed out {self.attempts} times - showing the last "
                f"snapshot for {self.key[0]} (as of {when}); data may be stale."
            )
        if self.cached:
            return f"📦 {self.key[0]} quotes served from cache (as of {when})."
        return f"✅ {self.key[0]} live quotes (as of {when})."


def default_raw_fetcher(ticker: str, start, end, adjust: str):
    """The single place that talks to Yahoo Finance."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    kwargs = {"start": pd.Timestamp(start), "end": pd.Timestamp(end)}
    actions = ADJUST_ACTIONS[str(adjust).lower()]
    if actions is not None:
        kwargs["actions"] = actions
    try:
        data = t.history(**kwargs, timeout=15)
    except TypeError:
        data = t.history(**kwargs)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return data, info


RawFetcher = Callable[[str, pd.Timestamp, pd.Timestamp, str], Tuple[pd.DataFrame, dict]]


class DataFetcher:
    """Fetching layer with LRU quote cache, per-key request coalescing and
    stale-snapshot fallback after three failed attempts.

    Cache isolation is by the full ``(ticker, start, end, adjust)`` key, so
    concurrent users looking at different stocks/ranges never share quotes.
    """

    def __init__(
        self,
        raw_fetcher: Optional[RawFetcher] = None,
        maxsize: int = 64,
        max_attempts: int = 3,
        retry_sleep: float = 0.5,
    ):
        self._raw = raw_fetcher or default_raw_fetcher
        self._cache = LRUCache(maxsize=maxsize)
        self._snapshots = {}  # ticker -> FetchResult of latest live success
        self._key_locks = {}
        self._guard = threading.RLock()
        self.max_attempts = max_attempts
        self.retry_sleep = retry_sleep

    @property
    def cache(self) -> LRUCache:
        return self._cache

    def _lock_for(self, key):
        with self._guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    @staticmethod
    def _normalize(data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.drop(columns=[c for c in ("Dividends", "Stock Splits", "Capital Gains")
                              if c in df.columns], errors="ignore")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df

    def _fresh_result(self, key, data, info) -> "FetchResult":
        result = FetchResult(
            data=data,
            info=info or {},
            key=key,
            attempts=1,
            fetched_at=time.time(),
            as_of=pd.Timestamp(data.index[-1]).tz_localize(None),
        )
        self._cache.set(key, result)
        self._snapshots[key[0]] = result
        return result

    def _stale_result(self, key, snapshot, error, attempts) -> "FetchResult":
        return FetchResult(
            data=snapshot.data,
            info=snapshot.info,
            key=key,
            stale=True,
            attempts=attempts,
            fetched_at=snapshot.fetched_at,
            as_of=snapshot.as_of,
            error=error,
        )

    def _attempt_fetch(self, key) -> "FetchResult":
        ticker, start, end, adjust = key
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                data, info = self._raw(ticker, start, end, adjust)
                data = self._normalize(data)
                if data is None or data.empty:
                    raise ValueError("empty quote returned")
                return self._fresh_result(key, data, info)
            except Exception as exc:  # timeouts / network / empty frames
                last_error = exc
                if attempt < self.max_attempts and self.retry_sleep:
                    time.sleep(self.retry_sleep)

        snapshot = self._snapshots.get(ticker)
        if snapshot is not None:
            return self._stale_result(key, snapshot, str(last_error), self.max_attempts)
        return FetchResult(
            data=None,
            info={},
            key=key,
            attempts=self.max_attempts,
            error=str(last_error),
        )

    @staticmethod
    def _cached_copy(key, cached: "FetchResult") -> "FetchResult":
        return FetchResult(
            data=cached.data,
            info=cached.info,
            key=key,
            cached=True,
            attempts=0,
            fetched_at=cached.fetched_at,
            as_of=cached.as_of,
            error=cached.error,
        )

    def fetch(self, ticker: str, start, end, adjust: str = "raw") -> "FetchResult":
        """Return quotes for one triple, coalescing concurrent duplicates."""
        key = normalize_key(ticker, start, end, adjust)

        cached = self._cache.get(key)
        if cached is not None:
            return self._cached_copy(key, cached)

        lock = self._lock_for(key)
        with lock:
            cached = self._cache.get(key)
            if cached is not None:
                return self._cached_copy(key, cached)
            return self._attempt_fetch(key)

    def clear(self, ticker: Optional[str] = None) -> None:
        """Drop cached quotes (and snapshot). No ticker clears everything."""
        if ticker is None:
            self._cache.clear()
            self._snapshots.clear()
            return
        t = ticker.strip().upper()
        with self._guard:
            for key in self._cache.keys():
                if key[0] == t:
                    self._cache.discard(key)
            self._snapshots.pop(t, None)


def sort_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV rows sorted by date index (deduplicated)."""
    df = data.copy()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Sort by date first, then compute indicators.

    Every windowed indicator uses ``min_periods=window``; rows where the
    window is incomplete stay ``NaN`` rather than becoming partial values
    or zeroes.
    """
    df = sort_ohlcv(data)
    close = df["Close"]

    # Moving averages (explicit min_periods -> NaN until the window is full)
    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()

    # MACD: signal is a 9-period EWM seeded on the first valid MACD row, so
    # it too is NaN for the first 8 MACD observations.
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = np.nan
    df["MACD_histogram"] = np.nan
    macd_valid = df["MACD"].dropna()
    if len(macd_valid) >= 9:
        signal = macd_valid.ewm(span=9, adjust=False, min_periods=9).mean()
        df.loc[signal.index, "MACD_signal"] = signal
        df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # RSI (Wilder smoothing): first 14 rows are NaN; zero losses -> 100.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    df["RSI"] = rsi

    # Bollinger Bands
    df["BB_middle"] = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std()
    df["BB_upper"] = df["BB_middle"] + bb_std * 2
    df["BB_lower"] = df["BB_middle"] - bb_std * 2

    # Volume
    df["Volume_SMA"] = df["Volume"].rolling(window=20, min_periods=20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"]

    # Point-in-time price structure (no look-back, valid from the first row)
    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["Price_Change"] = df["Close"] - df["Open"]
    df["Price_Change_Pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    # Average True Range
    df["High_Low"] = df["High"] - df["Low"]
    df["High_Close"] = (df["High"] - close.shift(1)).abs()
    df["Low_Close"] = (df["Low"] - close.shift(1)).abs()
    df["True_Range"] = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1)
    df["ATR"] = df["True_Range"].rolling(window=14, min_periods=14).mean()

    # Stochastic oscillator
    low_14 = df["Low"].rolling(window=14, min_periods=14).min()
    high_14 = df["High"].rolling(window=14, min_periods=14).max()
    denom = high_14 - low_14
    df["Stoch_K"] = 100.0 * (df["Close"] - low_14) / denom
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3, min_periods=3).mean()

    return df


def prepare_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features on top of the indicator layer (same sort/NaN rules)."""
    df = calculate_technical_indicators(data)

    df["Returns"] = df["Close"].pct_change()
    df["Returns_5d"] = df["Close"].pct_change(5)
    df["Returns_10d"] = df["Close"].pct_change(10)

    for lag in (1, 2, 3, 5, 10):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)
        df[f"Returns_lag_{lag}"] = df["Returns"].shift(lag)

    for window in (5, 10, 20, 50):
        df[f"Close_mean_{window}"] = df["Close"].rolling(window, min_periods=window).mean()
        df[f"Close_std_{window}"] = df["Close"].rolling(window, min_periods=window).std()
        df[f"Volume_mean_{window}"] = df["Volume"].rolling(window, min_periods=window).mean()
        df[f"High_mean_{window}"] = df["High"].rolling(window, min_periods=window).mean()
        df[f"Low_mean_{window}"] = df["Low"].rolling(window, min_periods=window).mean()

    df["Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"] * 100
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"] * 100

    df["Price_volatility_10d"] = df["Returns"].rolling(10, min_periods=10).std()
    df["Price_volatility_20d"] = df["Returns"].rolling(20, min_periods=20).std()

    return df


DEFAULT_FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "ATR", "Stoch_K", "Stoch_D",
    "Price_vs_SMA20", "Price_vs_SMA50",
    "Price_volatility_10d", "Price_volatility_20d",
    "Returns_lag_1", "Returns_lag_2", "Returns_lag_3", "Returns_lag_5", "Returns_lag_10",
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5", "Close_lag_10",
    "Volume_lag_1", "Volume_lag_2", "Volume_lag_3",
    "Close_mean_5", "Close_mean_10", "Close_mean_20",
    "Close_std_5", "Close_std_10", "Close_std_20",
    "Volume_mean_5", "Volume_mean_20",
]


def make_label_target(close: pd.Series) -> pd.Series:
    """Forward label: next period's close (shift(-1)).

    The only NaN introduced is the final bar (no tomorrow to learn); the
    label side therefore drops only the tail, never earlier warm-up rows.
    """
    return close.shift(-1)


def forward_returns(close: pd.Series) -> pd.Series:
    """Return earned from t to t+1: ``close[t+1]/close[t] - 1``.

    Same shift direction as :func:`make_label_target`; NaN only at the
    tail. Backtests drop these NaNs strictly (no zero-filling).
    """
    return close.shift(-1) / close - 1.0


def align_features_target(features: pd.DataFrame, target: pd.Series):
    """Align features and the forward target without look-ahead.

    Rows with non-ready *features* (warm-up head) are dropped; the label
    contributes only the single tail row. No ffill/bfill across time.
    """
    feat = features.copy()
    tgt = target.reindex(feat.index)

    feature_ready = ~feat.isna().any(axis=1)
    target_ready = ~tgt.isna()
    keep = feature_ready & target_ready

    dropped_tail = int((feature_ready & ~target_ready).sum())
    return feat.loc[keep], tgt.loc[keep], dropped_tail


def run_backtest(close: pd.Series, position: pd.Series) -> pd.DataFrame:
    """Run a strictly-aligned backtest.

    ``position`` at bar t is a signal known at the close of t (in [-1, 1]);
    it earns :func:`forward_returns` from t to t+1. Both series are sorted,
    reindexed and dropna'd strictly, so the result never contains the last
    bar (unknown return) and every row's index matches its position.
    """
    c = close.sort_index().copy()
    fwd = forward_returns(c)
    pos = position.reindex(c.index).sort_index()

    bt = pd.DataFrame({"close": c, "position": pos, "forward_return": fwd})
    bt = bt.dropna(subset=["position", "forward_return"])
    if bt.empty:
        bt["strategy_return"] = pd.Series(dtype=float)
        bt["equity"] = pd.Series(dtype=float)
        bt["benchmark"] = pd.Series(dtype=float)
        return bt

    bt["strategy_return"] = bt["position"] * bt["forward_return"]
    bt["equity"] = (1.0 + bt["strategy_return"]).cumprod()
    bt["benchmark"] = (1.0 + bt["forward_return"]).cumprod()
    return bt
