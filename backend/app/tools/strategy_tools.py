# backend/app/tools/strategy_tools.py
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
except Exception:
    ta = None


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # Normalize columns to standard names
    rename = {}
    for key in ["open", "high", "low", "close", "volume"]:
        if key in cols:
            rename[cols[key]] = key.capitalize()
    if rename:
        df = df.rename(columns=rename)
    return df


def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = ensure_ohlc(df)
    if ta:
        return ta.rsi(df["Close"], length=length)
    # Fallback RSI if pandas_ta not available
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    df = ensure_ohlc(df)
    if ta:
        macd_df = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        return macd_df
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD_12_26_9": macd_line, "MACDs_12_26_9": signal_line, "MACDh_12_26_9": hist})


def moving_averages(df: pd.DataFrame, short=20, long=50):
    df = ensure_ohlc(df)
    return pd.DataFrame({
        f"SMA_{short}": df["Close"].rolling(short).mean(),
        f"SMA_{long}": df["Close"].rolling(long).mean(),
    })


def ichimoku(df: pd.DataFrame):
    df = ensure_ohlc(df)
    high_9 = df["High"].rolling(9).max()
    low_9 = df["Low"].rolling(9).min()
    conversion = (high_9 + low_9) / 2

    high_26 = df["High"].rolling(26).max()
    low_26 = df["Low"].rolling(26).min()
    base = (high_26 + low_26) / 2

    span_a = ((conversion + base) / 2).shift(26)
    high_52 = df["High"].rolling(52).max()
    low_52 = df["Low"].rolling(52).min()
    span_b = ((high_52 + low_52) / 2).shift(26)
    lagging = df["Close"].shift(-26)

    return pd.DataFrame({
        "tenkan": conversion,
        "kijun": base,
        "senkou_a": span_a,
        "senkou_b": span_b,
        "chikou": lagging,
    })


def summarize_indicators(df: pd.DataFrame) -> dict:
    rsi_series = rsi(df)
    macd_df = macd(df)
    ma_df = moving_averages(df)
    ichi_df = ichimoku(df)

    latest = {
        "rsi": float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None,
        "macd": {
            "macd": float(macd_df.dropna().iloc[-1][0]) if not macd_df.dropna().empty else None,
            "signal": float(macd_df.dropna().iloc[-1][1]) if not macd_df.dropna().empty else None,
            "hist": float(macd_df.dropna().iloc[-1][2]) if not macd_df.dropna().empty else None,
        },
        "ma": {
            key: float(val) if val == val else None  # NaN guard
            for key, val in ma_df.dropna().iloc[-1].to_dict().items()
        } if not ma_df.dropna().empty else None,
        "ichimoku": {
            key: float(val) if val == val else None
            for key, val in ichi_df.dropna().iloc[-1].to_dict().items()
        } if not ichi_df.dropna().empty else None,
    }
    return latest


def swing_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []
    if ind["rsi"] is not None:
        if ind["rsi"] < 30:
            bullets.append("RSI indicates oversold; potential swing long if price confirms.")
        elif ind["rsi"] > 70:
            bullets.append("RSI indicates overbought; look for mean reversion short setups.")
        else:
            bullets.append("RSI neutral; wait for momentum confirmation.")
    if ind["macd"]["hist"] is not None:
        if ind["macd"]["hist"] > 0:
            bullets.append("MACD histogram positive; bullish momentum building.")
        else:
            bullets.append("MACD histogram negative; bearish momentum risk.")
    if ind["ma"]:
        ma = ind["ma"]
        keys = list(ma.keys())
        if len(keys) >= 2:
            short = ma[keys[0]]
            long = ma[keys[1]]
            if short and long:
                if short > long:
                    bullets.append("Short MA above long MA; trend bias up.")
                else:
                    bullets.append("Short MA below long MA; trend bias down.")
    return "\n".join(f"• {b}" for b in bullets) or "No strong swing signals."

def scalping_strategy(stock: str) -> str:
    """
    Advanced scalping strategy:
    Uses 5-minute candles with RSI, MACD, and Stochastics.
    """
    df = yf.download(stock, period="5d", interval="5m")
    if df.empty:
        return f"No intraday data found for {stock}."

    # --- RSI ---
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- Stochastic Oscillator (14, 3, 3) ---
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["%K"] = 100 * ((df["Close"] - low14) / (high14 - low14))
    df["%D"] = df["%K"].rolling(3).mean()

    latest = df.iloc[-1]

    # --- Scalping logic ---
    signals = []

    # RSI
    if latest["RSI"] < 30:
        signals.append("RSI oversold (possible BUY)")
    elif latest["RSI"] > 70:
        signals.append("RSI overbought (possible SELL)")

    # MACD
    if latest["MACD"] > latest["Signal"]:
        signals.append("MACD bullish crossover (BUY)")
    elif latest["MACD"] < latest["Signal"]:
        signals.append("MACD bearish crossover (SELL)")

    # Stochastic
    if latest["%K"] > latest["%D"] and latest["%K"] < 20:
        signals.append("Stochastic bullish reversal (BUY)")
    elif latest["%K"] < latest["%D"] and latest["%K"] > 80:
        signals.append("Stochastic bearish reversal (SELL)")

    if not signals:
        return f"Scalping Strategy for {stock}: No clear setup, stay flat."

    return f"Scalping Strategy for {stock}:\n- " + "\n- ".join(signals)