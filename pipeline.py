"""Data pipeline for the AI stock dashboard.

Data concerns stay separate from UI concerns:

* QuoteFetcher handles network access with an explicit, bounded LRU cache
  keyed by ``(ticker, start, end, adjust)`` so two different stocks or two
  different date ranges can never share a quote frame.
* Pure pandas/numpy functions turn raw OHLCV frames into indicators, ML
  labels and backtest returns.

Indicators are computed on date-sorted frames; windows that are too short
stay ``NaN`` (never 0).  ML labels use forward ``shift(-1)`` and only the
label-less tail is dropped, while backtest series are strictly aligned
after ``dropna`` so no look-ahead information leaks into the model.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# yfinance expression of the three supported adjustment modes:
# qfq -> split + dividend adjusted (forward adjusted)
# hfq -> back adjusted off the most recent close
# none -> raw historical prices
_ADJUST_YF = {
    "qfq": {"auto_adjust": True, "back_adjust": False},
    "hfq": {"auto_adjust": False, "back_adjust": True},
    "none": {"auto_adjust": False, "back_adjust": False},
    "raw": {"auto_adjust": False, "back_adjust": False},
}

VALID_ADJUSTMENTS = ("qfq", "hfq", "none")

FETCH_ATTEMPTS = 3
FETCH_TIMEOUT = 10


@dataclass
class QuoteSnapshot:
    """A single cached market-data payload.

    ``stale`` marks frames served from cache after failed live fetches;
    ``fetched_at`` is the wall-clock timestamp used for the freshness banner.
    """

    ticker: str
    start: str
    end: str
    adjust: str
    data: pd.DataFrame
    info: dict[str, Any]
    fetched_at: float
    stale: bool = False
    error: Optional[str] = None

    def fetched_at_display(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.fetched_at))


class LRUQuoteCache:
    """Thread-safe bounded LRU cache keyed by (ticker, start, end, adjust)."""

    def __init__(self, maxsize: int = 64):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._store: OrderedDict[tuple, QuoteSnapshot] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def make_key(ticker: str, start: str, end: str, adjust: str) -> tuple:
        return (str(ticker).strip().upper(), str(start), str(end), str(adjust))

    def get(self, key: tuple) -> Optional[QuoteSnapshot]:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: tuple, snapshot: QuoteSnapshot) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = snapshot
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class QuoteFetcher:
    """Fetch OHLCV quotes with retry, snapshot fallback and single-flight."""

    def __init__(self, cache: Optional[LRUQuoteCache] = None, maxsize: int = 64):
        self.cache = cache or LRUQuoteCache(maxsize=maxsize)
        self._key_locks_guard = threading.Lock()
        self._key_locks: dict[tuple, threading.Lock] = {}
        # Company metadata repeats for every date window of the same ticker;
        # cache it separately (keyed by ticker only) so switching ranges on
        # one stock never re-issues the info request.
        self._info_cache: dict[str, dict[str, Any]] = {}
        self._info_lock = threading.Lock()

    def clear_cache(self) -> None:
        self.cache.clear()
        with self._info_lock:
            self._info_cache.clear()

    def _key_lock(self, key: tuple) -> threading.Lock:
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    @staticmethod
    def _download(ticker: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        kwargs = _ADJUST_YF[adjust]
        return yf.download(
            ticker,
            start=start or None,
            end=end or None,
            interval="1d",
            auto_adjust=kwargs["auto_adjust"],
            back_adjust=kwargs["back_adjust"],
            progress=False,
            timeout=FETCH_TIMEOUT,
            multi_level_index=False,
            threads=False,
        )

    def _live_download(self, ticker: str, start: str, end: str, adjust: str):
        """Run up to FETCH_ATTEMPTS attempts; return (data, error)."""
        last_error: Optional[Exception] = None
        for attempt in range(1, FETCH_ATTEMPTS + 1):
            try:
                frame = self._download(ticker, start, end, adjust)
                if frame is not None and not frame.empty:
                    frame = self._normalize_frame(frame)
                    if not frame.empty:
                        return frame, None
                last_error = RuntimeError("download returned no rows")
            except Exception as exc:  # network/timeout/provider failures
                last_error = exc
            time.sleep(min(0.5 * attempt, 1.0))
        return None, last_error

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep canonical OHLCV columns on a sorted DatetimeIndex."""
        if isinstance(frame.columns, pd.MultiIndex):  # defensive for old yf
            frame.columns = [
                col[0] if isinstance(col, tuple) else col for col in frame.columns
            ]
        missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
        if missing:
            raise RuntimeError(f"download missing columns: {missing}")
        frame = frame[OHLCV_COLUMNS].copy()
        frame.index = pd.to_datetime(frame.index)
        return sort_ohlcv(frame)

    def _fetch_info(self, ticker: str) -> dict[str, Any]:
        with self._info_lock:
            cached = self._info_cache.get(ticker)
            if cached is not None:
                return cached
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        if info:  # only cache usable responses
            with self._info_lock:
                self._info_cache[ticker] = info
        return info

    def fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        adjust: str = "qfq",
        force_refresh: bool = False,
    ) -> QuoteSnapshot:
        """Return a snapshot for the key, refreshing the cache when needed.

        After three failed attempts the most recent snapshot for this exact
        key is returned with ``stale=True``.  With no prior snapshot the
        last exception is re-raised so callers can show an error.
        """
        if adjust not in _ADJUST_YF:
            raise ValueError(f"unsupported adjust mode: {adjust!r}")
        key = self.cache.make_key(ticker, start, end, adjust)
        ticker_norm = key[0]

        with self._key_lock(key):  # single-flight for identical tuple keys
            if not force_refresh:
                cached = self.cache.get(key)
                if cached is not None:
                    return cached

            data, error = self._live_download(ticker_norm, start, end, adjust)

            if data is not None:
                snapshot = QuoteSnapshot(
                    ticker=ticker_norm,
                    start=start,
                    end=end,
                    adjust=adjust,
                    data=data,
                    info=self._fetch_info(ticker_norm),
                    fetched_at=time.time(),
                    stale=False,
                )
                self.cache.put(key, snapshot)
                return snapshot

            cached = self.cache.get(key)
            if cached is not None:
                return QuoteSnapshot(
                    ticker=cached.ticker,
                    start=cached.start,
                    end=cached.end,
                    adjust=cached.adjust,
                    data=cached.data,
                    info=cached.info,
                    fetched_at=cached.fetched_at,
                    stale=True,
                    error=str(error) if error is not None else "fetch failed",
                )

            raise RuntimeError(
                f"failed to fetch {ticker_norm} after "
                f"{FETCH_ATTEMPTS} attempts: {error}"
            )


# All Streamlit sessions share one process-wide fetcher; the tuple key keeps
# every stock/date/adjust combination fully isolated.
default_fetcher = QuoteFetcher()


# ---------------------------------------------------------------------------
# Indicator layer (pure functions)
# ---------------------------------------------------------------------------


def sort_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and de-duplicate out-of-order timestamps."""
    frame = data.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.sort_index()


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators with full-window warm-up (NaN, never 0).

    Dates are sorted first.  Every rolling/EWM window uses ``min_periods``
    equal to its span so insufficient history stays empty.
    """
    df = sort_ohlcv(data)
    close = df["Close"]

    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()

    # MACD only exists after EMA_26 warms up; signal needs 9 MACD values.
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "RSI"] = 100.0
    df.loc[(loss == 0) & (gain == 0), "RSI"] = 50.0

    df["BB_middle"] = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std()
    df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
    df["BB_lower"] = df["BB_middle"] - (bb_std * 2)

    df["Volume_SMA"] = df["Volume"].rolling(window=20, min_periods=20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"]

    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["Price_Change"] = df["Close"] - df["Open"]
    df["Price_Change_Pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    prev_close = close.shift(1)
    df["High_Low"] = df["High"] - df["Low"]
    df["High_Close"] = (df["High"] - prev_close).abs()
    df["Low_Close"] = (df["Low"] - prev_close).abs()
    df["True_Range"] = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1)
    df["ATR"] = df["True_Range"].rolling(window=14, min_periods=14).mean()

    low_14 = df["Low"].rolling(window=14, min_periods=14).min()
    high_14 = df["High"].rolling(window=14, min_periods=14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    df["Stoch_K"] = 100 * ((close - low_14) / denom)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3, min_periods=3).mean()

    return df


def prepare_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features using only backward-looking shifts/rolls."""
    df = data.copy()
    close = df["Close"]

    df["Returns"] = close.pct_change()
    df["Returns_5d"] = close.pct_change(5)
    df["Returns_10d"] = close.pct_change(10)

    for lag in (1, 2, 3, 5, 10):
        df[f"Close_lag_{lag}"] = close.shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)
        df[f"Returns_lag_{lag}"] = df["Returns"].shift(lag)

    for window in (5, 10, 20, 50):
        df[f"Close_mean_{window}"] = close.rolling(
            window, min_periods=window
        ).mean()
        df[f"Close_std_{window}"] = close.rolling(
            window, min_periods=window
        ).std()
        df[f"Volume_mean_{window}"] = df["Volume"].rolling(
            window, min_periods=window
        ).mean()
        df[f"High_mean_{window}"] = df["High"].rolling(
            window, min_periods=window
        ).mean()
        df[f"Low_mean_{window}"] = df["Low"].rolling(
            window, min_periods=window
        ).mean()

    df["Price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"] * 100
    df["Price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"] * 100

    df["Price_volatility_10d"] = df["Returns"].rolling(
        10, min_periods=10
    ).std()
    df["Price_volatility_20d"] = df["Returns"].rolling(
        20, min_periods=20
    ).std()

    return df


# Columns that are safe model inputs (no raw price/volume leakage).
FEATURE_COLUMNS = [
    "RSI",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "ATR",
    "Stoch_K",
    "Stoch_D",
    "Volume_ratio",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "Price_volatility_10d",
    "Price_volatility_20d",
] + [
    f"Close_lag_{lag}" for lag in (1, 2, 3, 5, 10)
] + [
    f"Volume_lag_{lag}" for lag in (1, 2, 3, 5, 10)
] + [
    f"Returns_lag_{lag}" for lag in (1, 2, 3, 5, 10)
] + [
    f"Close_mean_{w}" for w in (5, 10, 20, 50)
] + [
    f"Close_std_{w}" for w in (5, 10, 20, 50)
] + [
    f"Volume_mean_{w}" for w in (5, 10, 20, 50)
] + [
    f"High_mean_{w}" for w in (5, 10, 20, 50)
] + [
    f"Low_mean_{w}" for w in (5, 10, 20, 50)
]


def make_prediction_label(data: pd.DataFrame) -> pd.Series:
    """Forward label: next session's close (shift direction = -1)."""
    return data["Close"].shift(-1)


def build_prediction_dataset(
    data: pd.DataFrame, feature_columns: Optional[list[str]] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features with the forward label without look-ahead leakage.

    Alignment order matters:
    1. label is created with ``shift(-1)`` on the *full* frame, so the final
       bar has NaN label and is the only tail row dropped;
    2. warm-up rows whose features need more history are dropped at the head
       via ``dropna`` on features.  Forward fills are never used to fake
       history, and no future column is ever shifted backward.
    """
    ordered = sort_ohlcv(data)
    features_all = prepare_ml_features(ordered)
    cols = [c for c in (feature_columns or FEATURE_COLUMNS) if c in features_all]
    x = features_all[cols]
    y = make_prediction_label(ordered)
    y.name = "Target_Next_Close"

    valid_x = x.dropna()
    y = y.loc[valid_x.index]
    valid_y = y.dropna()  # tail-only removal: last bar has no next close
    x = valid_x.loc[valid_y.index]
    return x, valid_y


def forward_returns(data: pd.DataFrame) -> pd.Series:
    """Return earned by buying at bar t's close and selling at t+1 close."""
    close = sort_ohlcv(data)["Close"]
    return close.shift(-1) / close - 1


def backtest_returns(
    data: pd.DataFrame, signals: pd.Series
) -> pd.DataFrame:
    """Align a signal with strictly dropna'd forward returns.

    The returned frame is indexed by *entry* date and shares one index across
    ``Forward_Return``, ``Signal`` and ``Strategy_Return``.  Both the head
    (missing signal) and the tail (no exit price yet) are dropped, so the
    first/last rows align exactly.
    """
    ordered = sort_ohlcv(data)
    fwd = forward_returns(ordered)
    sig = signals.reindex(ordered.index).astype(float)

    frame = pd.DataFrame(
        {"Forward_Return": fwd, "Signal": sig}, index=ordered.index
    )
    frame = frame.dropna()  # strict: head NaN signal + tail NaN return go
    frame["Strategy_Return"] = frame["Signal"] * frame["Forward_Return"]
    return frame
