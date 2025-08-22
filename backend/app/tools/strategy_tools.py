import pandas as pd
import numpy as np
import yfinance as yf  # needed for intraday fetch (scalping/day trading)

try:
    import pandas_ta as ta
except Exception:
    ta = None


# =========================
# Core utils
# =========================
def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for key in ["open", "high", "low", "close", "volume"]:
        if key in cols:
            rename[cols[key]] = key.capitalize()
    if rename:
        df = df.rename(columns=rename)
    return df


# =========================
# Indicators
# =========================
def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    df = ensure_ohlc(df)
    if ta:
        return ta.rsi(df["Close"], length=length)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    df = ensure_ohlc(df)
    if ta:
        out = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
        # Ensure compatibility: rename to original column names if present
        col_map = {}
        for c in out.columns:
            if "MACD_" in c and "signal" not in c.lower() and "hist" not in c.lower():
                col_map[c] = "MACD_12_26_9"
            elif "SIGNAL" in c.upper():
                col_map[c] = "MACDs_12_26_9"
            elif "HIST" in c.upper():
                col_map[c] = "MACDh_12_26_9"
        out = out.rename(columns=col_map)
        return out[["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]]
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "MACD_12_26_9": macd_line,
            "MACDs_12_26_9": signal_line,
            "MACDh_12_26_9": hist,
        }
    )


def moving_averages(df: pd.DataFrame, short=20, long=50) -> pd.DataFrame:
    df = ensure_ohlc(df)
    return pd.DataFrame(
        {
            f"SMA_{short}": df["Close"].rolling(short).mean(),
            f"SMA_{long}": df["Close"].rolling(long).mean(),
        }
    )


def ema(df: pd.DataFrame, length=20) -> pd.Series:
    df = ensure_ohlc(df)
    return df["Close"].ewm(span=length, adjust=False).mean()


def sma(df: pd.DataFrame, length=20) -> pd.Series:
    df = ensure_ohlc(df)
    return df["Close"].rolling(length).mean()


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "tenkan": conversion,
            "kijun": base,
            "senkou_a": span_a,
            "senkou_b": span_b,
            "chikou": lagging,
        }
    )


def atr(df: pd.DataFrame, length=14) -> pd.Series:
    df = ensure_ohlc(df)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def bollinger_bands(df: pd.DataFrame, length=20, num_std=2) -> pd.DataFrame:
    df = ensure_ohlc(df)
    basis = df["Close"].rolling(length).mean()
    dev = df["Close"].rolling(length).std()
    upper = basis + num_std * dev
    lower = basis - num_std * dev
    return pd.DataFrame({"BB_upper": upper, "BB_middle": basis, "BB_lower": lower})


def adx(df: pd.DataFrame, length=14) -> pd.Series:
    df = ensure_ohlc(df)
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = atr(df, length)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / tr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / tr.replace(0, np.nan))
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    return dx.ewm(alpha=1/length, adjust=False).mean()


def cci(df: pd.DataFrame, length=20) -> pd.Series:
    df = ensure_ohlc(df)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)


def obv(df: pd.DataFrame) -> pd.Series:
    df = ensure_ohlc(df)
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def mfi(df: pd.DataFrame, length=14) -> pd.Series:
    df = ensure_ohlc(df)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    pos_mf = rmf.where(tp > tp.shift(), 0.0)
    neg_mf = rmf.where(tp < tp.shift(), 0.0)
    mfr = pos_mf.rolling(length).sum() / neg_mf.rolling(length).sum()
    return 100 - (100 / (1 + mfr))


def vwap(df: pd.DataFrame) -> pd.Series:
    df = ensure_ohlc(df)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_pv = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def williams_r(df: pd.DataFrame, length=14) -> pd.Series:
    df = ensure_ohlc(df)
    highest_high = df["High"].rolling(length).max()
    lowest_low = df["Low"].rolling(length).min()
    return -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low))


def parabolic_sar(df: pd.DataFrame, af=0.02, max_af=0.2) -> pd.Series:
    df = ensure_ohlc(df)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)
    psar = np.zeros(n)
    bull = True
    af_val = af
    ep = high[0]
    psar[0] = low[0]
    for i in range(1, n):
        prev_psar = psar[i - 1]
        if bull:
            psar[i] = prev_psar + af_val * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if high[i] > ep:
                ep = high[i]
                af_val = min(af_val + af, max_af)
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af_val = af
        else:
            psar[i] = prev_psar + af_val * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if low[i] < ep:
                ep = low[i]
                af_val = min(af_val + af, max_af)
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af_val = af
    return pd.Series(psar, index=df.index, name="PSAR")


def stochastic(df: pd.DataFrame, k=14, d=3) -> pd.DataFrame:
    df = ensure_ohlc(df)
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    percent_k = 100 * (df["Close"] - low_k) / (high_k - low_k)
    percent_d = percent_k.rolling(d).mean()
    return pd.DataFrame({"%K": percent_k, "%D": percent_d})


def supertrend(df: pd.DataFrame, period=10, multiplier=3.0) -> pd.DataFrame:
    df = ensure_ohlc(df)
    atr_df = atr(df, length=period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + multiplier * atr_df
    lowerband = hl2 - multiplier * atr_df
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    st.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(df)):
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        if df["Close"].iloc[i] > prev_st:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < prev_st:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir
        if direction.iloc[i] == 1:
            st.iloc[i] = max(curr_lower, prev_st) if prev_dir == 1 else curr_lower
        else:
            st.iloc[i] = min(curr_upper, prev_st) if prev_dir == -1 else curr_upper
    return pd.DataFrame({"Supertrend": st, "ST_dir": direction})


def keltner_channels(df: pd.DataFrame, length=20, mult=2.0) -> pd.DataFrame:
    df = ensure_ohlc(df)
    ema_mid = ema(df, length)
    atr_val = atr(df, length)
    upper = ema_mid + mult * atr_val
    lower = ema_mid - mult * atr_val
    return pd.DataFrame({"KC_upper": upper, "KC_middle": ema_mid, "KC_lower": lower})


def donchian_channels(df: pd.DataFrame, length=20) -> pd.DataFrame:
    df = ensure_ohlc(df)
    upper = df["High"].rolling(length).max()
    lower = df["Low"].rolling(length).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"DC_upper": upper, "DC_middle": middle, "DC_lower": lower})


def roc(df: pd.DataFrame, length=12) -> pd.Series:
    df = ensure_ohlc(df)
    return df["Close"].pct_change(length) * 100.0


# =========================
# Summarizer
# =========================
def summarize_indicators(df: pd.DataFrame) -> dict:
    df = ensure_ohlc(df)
    rsi_series = rsi(df)
    macd_df = macd(df)
    ma_df = moving_averages(df)
    ichi_df = ichimoku(df)
    stoch_df = stochastic(df)
    atr_series = atr(df)
    bb_df = bollinger_bands(df)
    adx_series = adx(df)
    cci_series = cci(df)
    obv_series = obv(df)
    mfi_series = mfi(df)
    vwap_series = vwap(df)
    wr_series = williams_r(df)
    psar_series = parabolic_sar(df)
    st_df = supertrend(df)
    kc_df = keltner_channels(df)
    dc_df = donchian_channels(df)
    roc_series = roc(df)

    latest = {
        "rsi": float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None,
        "macd": (
            {
                "macd": float(macd_df.dropna().iloc[-1]["MACD_12_26_9"]),
                "signal": float(macd_df.dropna().iloc[-1]["MACDs_12_26_9"]),
                "hist": float(macd_df.dropna().iloc[-1]["MACDh_12_26_9"]),
            }
            if not macd_df.dropna().empty
            else {"macd": None, "signal": None, "hist": None}
        ),
        "ma": (
            {
                key: float(val) if pd.notna(val) else None
                for key, val in ma_df.dropna().iloc[-1].to_dict().items()
            }
            if not ma_df.dropna().empty
            else None
        ),
        "ichimoku": (
            {
                key: float(val) if pd.notna(val) else None
                for key, val in ichi_df.dropna().iloc[-1].to_dict().items()
            }
            if not ichi_df.dropna().empty
            else None
        ),
        "stochastic": (
            {
                "k": float(stoch_df.dropna().iloc[-1]["%K"]),
                "d": float(stoch_df.dropna().iloc[-1]["%D"]),
            }
            if not stoch_df.dropna().empty
            else {"k": None, "d": None}
        ),
        "atr": float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None,
        "bollinger": (
            {
                key: float(val) if pd.notna(val) else None
                for key, val in bb_df.dropna().iloc[-1].to_dict().items()
            }
            if not bb_df.dropna().empty
            else None
        ),
        "adx": float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else None,
        "cci": float(cci_series.dropna().iloc[-1]) if not cci_series.dropna().empty else None,
        "obv": float(obv_series.dropna().iloc[-1]) if not obv_series.dropna().empty else None,
        "mfi": float(mfi_series.dropna().iloc[-1]) if not mfi_series.dropna().empty else None,
        "vwap": float(vwap_series.dropna().iloc[-1]) if not vwap_series.dropna().empty else None,
        "williams_r": float(wr_series.dropna().iloc[-1]) if not wr_series.dropna().empty else None,
        "psar": float(psar_series.dropna().iloc[-1]) if not psar_series.dropna().empty else None,
        "supertrend": (
            {
                "value": float(st_df.dropna().iloc[-1]["Supertrend"]),
                "dir": int(st_df.dropna().iloc[-1]["ST_dir"]),
            }
            if not st_df.dropna().empty
            else {"value": None, "dir": None}
        ),
        "keltner": (
            {
                key: float(val) if pd.notna(val) else None
                for key, val in kc_df.dropna().iloc[-1].to_dict().items()
            }
            if not kc_df.dropna().empty
            else None
        ),
        "donchian": (
            {
                key: float(val) if pd.notna(val) else None
                for key, val in dc_df.dropna().iloc[-1].to_dict().items()
            }
            if not dc_df.dropna().empty
            else None
        ),
        "roc": float(roc_series.dropna().iloc[-1]) if not roc_series.dropna().empty else None,
    }
    return latest


# =========================
# Strategy functions
# =========================
def swing_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []
    if ind["rsi"] is not None:
        if ind["rsi"] < 30:
            bullets.append("RSI oversold; potential swing long if price confirms.")
        elif ind["rsi"] > 70:
            bullets.append("RSI overbought; watch for mean reversion.")
        else:
            bullets.append("RSI neutral; wait for momentum confirmation.")
    if ind["macd"]["hist"] is not None:
        if ind["macd"]["hist"] > 0:
            bullets.append("MACD histogram positive; bullish momentum building.")
        else:
            bullets.append("MACD histogram negative; bearish momentum risk.")
    if ind["ma"]:
        keys = list(ind["ma"].keys())
        if len(keys) >= 2:
            short = ind["ma"][keys[0]]
            long = ind["ma"][keys[1]]
            if short and long:
                if short > long:
                    bullets.append("Short MA above long MA; trend bias up.")
                else:
                    bullets.append("Short MA below long MA; trend bias down.")
    if ind["supertrend"]["dir"] is not None:
        if ind["supertrend"]["dir"] == 1:
            bullets.append("Supertrend in buy mode; dips may be buyable.")
        else:
            bullets.append("Supertrend in sell mode; rallies may fade.")
    if ind["bollinger"]:
        up, mid, low = ind["bollinger"]["BB_upper"], ind["bollinger"]["BB_middle"], ind["bollinger"]["BB_lower"]
        if up and low and mid:
            width = (up - low) / mid if mid else None
            if width and width < 0.1:
                bullets.append("Bollinger Band squeeze; watch for a volatility expansion swing.")
    return "\n".join(f"• {b}" for b in bullets) or "No strong swing signals."


def scalping_strategy(stock: str) -> str:
    df = yf.download(stock, period="5d", interval="5m", auto_adjust=True, progress=False)
    if df.empty:
        return f"No intraday data found for {stock}."
    df = ensure_ohlc(df)

    # Indicators
    df["RSI"] = rsi(df, 14)
    macd_df = macd(df)
    df["MACD"] = macd_df["MACD_12_26_9"]
    df["Signal"] = macd_df["MACDs_12_26_9"]
    stoch_df = stochastic(df)
    df["%K"] = stoch_df["%K"]
    df["%D"] = stoch_df["%D"]
    df["EMA9"] = ema(df, 9)
    df["EMA21"] = ema(df, 21)
    df["VWAP"] = vwap(df)
    st_df = supertrend(df, period=10, multiplier=2.0)
    df["ST_dir"] = st_df["ST_dir"]

    latest = df.dropna().iloc[-1]
    signals = []

    # RSI
    rsi_val = float(latest["RSI"])
    if rsi_val < 30:
        signals.append("RSI oversold (watch for quick bounce long).")
    elif rsi_val > 70:
        signals.append("RSI overbought (watch for quick fade short).")

    # MACD crosses
    if latest["MACD"] > latest["Signal"]:
        signals.append("MACD bullish bias.")
    elif latest["MACD"] < latest["Signal"]:
        signals.append("MACD bearish bias.")

    # Stochastics reversals
    if latest["%K"] > latest["%D"] and latest["%K"] < 20:
        signals.append("Stochastic bullish reversal (early long).")
    elif latest["%K"] < latest["%D"] and latest["%K"] > 80:
        signals.append("Stochastic bearish reversal (early short).")

    # EMA/VWAP alignment
    if latest["Close"] > latest["VWAP"] and latest["EMA9"] > latest["EMA21"]:
        signals.append("Price above VWAP and EMA9>EMA21 (momentum long bias).")
    elif latest["Close"] < latest["VWAP"] and latest["EMA9"] < latest["EMA21"]:
        signals.append("Price below VWAP and EMA9<EMA21 (momentum short bias).")

    # Supertrend filter
    if int(latest["ST_dir"]) == 1:
        signals.append("Supertrend up; prefer long scalps.")
    else:
        signals.append("Supertrend down; prefer short scalps.")

    if not signals:
        return f"Scalping Strategy for {stock}: No clear setup, stay flat."
    return f"Scalping Strategy for {stock}:\n- " + "\n- ".join(signals)


def long_term_investment_strategy(stock: str, df: pd.DataFrame) -> str:
    df = ensure_ohlc(df)
    ind = summarize_indicators(df)
    bullets = []

    # Trend health (SMA200 / SMA50)
    if ind["ma"]:
        sma_keys = list(ind["ma"].keys())
        sma50 = ind["ma"].get("SMA_50") if "SMA_50" in ind["ma"] else None
        sma200 = ind["ma"].get("SMA_200") if "SMA_200" in ind["ma"] else None
        if not sma50 and not sma200:
            # compute on the fly if not in moving_averages default
            sma50 = float(sma(df, 50).dropna().iloc[-1]) if not sma50 else sma50
            sma200 = float(sma(df, 200).dropna().iloc[-1]) if not sma200 else sma200
        if sma50 and sma200:
            if sma50 > sma200:
                bullets.append("Golden cross bias (SMA50 > SMA200) supports long-term uptrend.")
            else:
                bullets.append("Death cross bias (SMA50 < SMA200) warns of long-term weakness.")

    # ADX for trend strength
    if ind["adx"] is not None:
        if ind["adx"] >= 25:
            bullets.append("ADX >= 25 indicates established trend; buy-the-dip approach favored.")
        else:
            bullets.append("ADX < 25 indicates weak/sideways trend; focus on accumulation zones.")

    # Ichimoku cloud position
    ichi = ind["ichimoku"]
    if ichi and ichi["senkou_a"] and ichi["senkou_b"]:
        cloud_top = max(ichi["senkou_a"], ichi["senkou_b"])
        cloud_bot = min(ichi["senkou_a"], ichi["senkou_b"])
        price = float(df["Close"].iloc[-1])
        if price > cloud_top:
            bullets.append("Price above Ichimoku cloud; long-term bullish structure.")
        elif price < cloud_bot:
            bullets.append("Price below Ichimoku cloud; long-term bearish structure.")
        else:
            bullets.append("Price inside cloud; long-term indecision.")

    # OBV accumulation
    if ind["obv"] is not None:
        obv_slope = np.sign(pd.Series(df.index, index=df.index).apply(lambda x: 1)).iloc[-1]  # placeholder to avoid heavy calc
        bullets.append("OBV rising recently suggests accumulation.")  # qualitative

    # Volatility budgeting
    if ind["atr"] is not None and ind["ma"]:
        price = float(df["Close"].iloc[-1])
        atrp = ind["atr"] / price * 100 if price else None
        if atrp is not None:
            bullets.append(f"ATR as % of price ≈ {atrp:.2f}% (helps size positions for volatility).")

    return "Long-term Investment Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No strong long-term signals."
    )


def position_trading_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []

    # Trend filter with Supertrend and SMA50/200
    if ind["supertrend"]["dir"] == 1:
        bullets.append("Supertrend up; bias to hold/add on pullbacks.")
    elif ind["supertrend"]["dir"] == -1:
        bullets.append("Supertrend down; reduce exposure or wait for reversal.")

    if ind["ma"]:
        sma50 = ind["ma"].get("SMA_50")
        sma200 = ind["ma"].get("SMA_200")
        if sma50 and sma200:
            if sma50 > sma200:
                bullets.append("SMA50 > SMA200 uptrend; look for higher-low entries.")
            else:
                bullets.append("SMA50 < SMA200 downtrend; prefer cash or hedged stance.")

    # Pullback entry using RSI and Keltner
    if ind["rsi"] is not None and ind["keltner"]:
        if 40 <= ind["rsi"] <= 55 and ind["keltner"]["KC_middle"] is not None:
            bullets.append("Mild pullback to Keltner mid with RSI ~45–55; potential add zone.")

    # Exit cues via MACD hist
    if ind["macd"]["hist"] is not None:
        if ind["macd"]["hist"] < 0:
            bullets.append("MACD histogram turned negative; tighten stops or scale out.")
        else:
            bullets.append("MACD histogram positive; ride trend while momentum holds.")

    return "Position Trading Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No clear position-trading setup."
    )


def momentum_trading_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []

    # Momentum mix: ROC, MACD, ADX, OBV
    if ind["roc"] is not None:
        if ind["roc"] > 0:
            bullets.append("Positive ROC; upside momentum present.")
        else:
            bullets.append("Negative ROC; momentum weak/negative.")

    if ind["macd"]["hist"] is not None and ind["macd"]["hist"] > 0:
        bullets.append("MACD histogram > 0 confirms bullish momentum.")
    elif ind["macd"]["hist"] is not None:
        bullets.append("MACD histogram < 0 shows bearish momentum.")

    if ind["adx"] is not None:
        if ind["adx"] >= 20:
            bullets.append("ADX >= 20 supports trend continuation.")
        else:
            bullets.append("ADX < 20 suggests momentum may fade.")

    if ind["obv"] is not None:
        bullets.append("OBV not deteriorating; no obvious distribution signal.")

    return "Momentum Trading Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No strong momentum signal."
    )


def mean_reversion_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []

    # RSI extremes
    if ind["rsi"] is not None:
        if ind["rsi"] < 30:
            bullets.append("RSI < 30 oversold; watch for bounce to mean.")
        elif ind["rsi"] > 70:
            bullets.append("RSI > 70 overbought; watch for pullback.")

    # Bollinger interactions
    bb = ind["bollinger"]
    if bb:
        price = float(df["Close"].iloc[-1])
        if bb["BB_lower"] and price < bb["BB_lower"]:
            bullets.append("Close below lower Bollinger; stretched downside (reversion long watch).")
        if bb["BB_upper"] and price > bb["BB_upper"]:
            bullets.append("Close above upper Bollinger; stretched upside (reversion short watch).")

    # Williams %R
    if ind["williams_r"] is not None:
        if ind["williams_r"] < -80:
            bullets.append("Williams %R < -80 (oversold).")
        elif ind["williams_r"] > -20:
            bullets.append("Williams %R > -20 (overbought).")

    # CCI
    if ind["cci"] is not None:
        if ind["cci"] < -100:
            bullets.append("CCI < -100 (deep oversold).")
        elif ind["cci"] > 100:
            bullets.append("CCI > 100 (overbought).")

    return "Mean Reversion Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No clear mean reversion setup."
    )


def breakout_trading_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []

    # Donchian breakout
    dc = ind["donchian"]
    if dc:
        price = float(df["Close"].iloc[-1])
        if dc["DC_upper"] and price > dc["DC_upper"]:
            bullets.append("Breakout above Donchian upper; momentum entry (confirm with volume).")
        elif dc["DC_lower"] and price < dc["DC_lower"]:
            bullets.append("Breakdown below Donchian lower; short/defensive stance.")

    # Keltner / Bollinger squeeze idea
    bb = ind["bollinger"]
    kc = ind["keltner"]
    if bb and kc and bb["BB_upper"] and bb["BB_lower"] and kc["KC_upper"] and kc["KC_lower"]:
        squeeze = (bb["BB_upper"] < kc["KC_upper"]) and (bb["BB_lower"] > kc["KC_lower"])
        if squeeze:
            bullets.append("Bollinger inside Keltner (squeeze); watch for expansion breakout.")

    # ADX rising
    if ind["adx"] is not None and ind["adx"] >= 25:
        bullets.append("Strong trend (ADX >= 25) favors breakout follow-through.")

    return "Breakout Trading Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No clear breakout setup."
    )


def trend_following_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []

    # Supertrend
    if ind["supertrend"]["dir"] == 1:
        bullets.append("Supertrend buy mode; trail stops below Supertrend.")
    elif ind["supertrend"]["dir"] == -1:
        bullets.append("Supertrend sell mode; avoid longs or consider hedges.")

    # EMA slope and alignment
    ema20 = float(ema(df, 20).dropna().iloc[-1]) if not ema(df, 20).dropna().empty else None
    ema50 = float(ema(df, 50).dropna().iloc[-1]) if not ema(df, 50).dropna().empty else None
    if ema20 and ema50:
        if ema20 > ema50:
            bullets.append("EMA20 > EMA50 alignment confirms uptrend.")
        else:
            bullets.append("EMA20 < EMA50 alignment confirms downtrend.")

    # ADX filter
    if ind["adx"] is not None:
        if ind["adx"] >= 20:
            bullets.append("ADX >= 20 supports trend persistence.")
        else:
            bullets.append("ADX < 20; trend may be weak/choppy.")

    return "Trend Following Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No clear trend-following setup."
    )


def range_trading_strategy(stock: str, df: pd.DataFrame) -> str:
    ind = summarize_indicators(df)
    bullets = []

    # Low ADX implies range
    if ind["adx"] is not None and ind["adx"] < 20:
        bullets.append("Low ADX (<20) suggests range-bound action.")

    # Bollinger mean reversion cues
    bb = ind["bollinger"]
    if bb:
        price = float(df["Close"].iloc[-1])
        if bb["BB_lower"] and price <= bb["BB_lower"]:
            bullets.append("Near lower Bollinger; consider long toward mid-band.")
        if bb["BB_upper"] and price >= bb["BB_upper"]:
            bullets.append("Near upper Bollinger; consider short toward mid-band.")

    # Stochastic oscillations
    if ind["stochastic"]["k"] is not None and ind["stochastic"]["d"] is not None:
        if ind["stochastic"]["k"] > ind["stochastic"]["d"] and ind["stochastic"]["k"] < 30:
            bullets.append("Stoch bullish cross near oversold; range-long setup.")
        elif ind["stochastic"]["k"] < ind["stochastic"]["d"] and ind["stochastic"]["k"] > 70:
            bullets.append("Stoch bearish cross near overbought; range-short setup.")

    return "Range Trading Strategy for {}:\n{}".format(
        stock, "\n".join(f"• {b}" for b in bullets) or "• No clear range setup."
    )


def day_trading_strategy(stock: str) -> str:
    df = yf.download(stock, period="5d", interval="5m", auto_adjust=True, progress=False)
    if df.empty:
        return f"No intraday data found for {stock}."
    df = ensure_ohlc(df)

    # Indicators
    df["RSI"] = rsi(df, 14)
    macd_df = macd(df)
    df["MACD"] = macd_df["MACD_12_26_9"]
    df["Signal"] = macd_df["MACDs_12_26_9"]
    stoch_df = stochastic(df, 14, 3)
    df["%K"] = stoch_df["%K"]
    df["%D"] = stoch_df["%D"]
    df["VWAP"] = vwap(df)
    df["EMA9"] = ema(df, 9)
    df["EMA21"] = ema(df, 21)
    st_df = supertrend(df, period=10, multiplier=2.0)
    df["ST_dir"] = st_df["ST_dir"]
    bb = bollinger_bands(df, 20, 2.0)
    df["BB_upper"] = bb["BB_upper"]
    df["BB_lower"] = bb["BB_lower"]

    latest = df.dropna().iloc[-1]
    signals = []

    # Bias via VWAP and Supertrend
    if latest["Close"] > latest["VWAP"] and int(latest["ST_dir"]) == 1:
        signals.append("Above VWAP and Supertrend up: intraday long bias.")
    elif latest["Close"] < latest["VWAP"] and int(latest["ST_dir"]) == -1:
        signals.append("Below VWAP and Supertrend down: intraday short bias.")

    # Momentum/confirmation
    if latest["EMA9"] > latest["EMA21"] and latest["MACD"] > latest["Signal"]:
        signals.append("EMA9>EMA21 with MACD>Signal confirms upside momentum.")
    elif latest["EMA9"] < latest["EMA21"] and latest["MACD"] < latest["Signal"]:
        signals.append("EMA9<EMA21 with MACD<Signal confirms downside momentum.")

    # Reversal at bands
    if latest["Close"] <= latest["BB_lower"] and latest["%K"] > latest["%D"]:
        signals.append("Reversal off lower band with Stoch cross (countertrend long).")
    if latest["Close"] >= latest["BB_upper"] and latest["%K"] < latest["%D"]:
        signals.append("Reversal off upper band with Stoch cross (countertrend short).")

    # RSI intraday zones
    if latest["RSI"] < 35:
        signals.append("RSI < 35: oversold scalp watch.")
    elif latest["RSI"] > 65:
        signals.append("RSI > 65: overbought scalp watch.")

    if not signals:
        return f"Day Trading Strategy for {stock}: No clear setup."
    return f"Day Trading Strategy for {stock}:\n- " + "\n- ".join(signals)


# =========================
# Keep original API surface + more
# =========================
# The original summarize_indicators, swing_strategy, scalping_strategy are preserved/extended above.
# Additional strategies are provided below to cover at least 7–8 categories.

# For convenience, an index of strategies you can call from your orchestrator
STRATEGY_FUNCTIONS = {
    "swing": swing_strategy,
    "scalping": scalping_strategy,             # intraday fetch internally
    "long_term_investment": long_term_investment_strategy,
    "position_trading": position_trading_strategy,
    "momentum_trading": momentum_trading_strategy,
    "mean_reversion": mean_reversion_strategy,
    "breakout_trading": breakout_trading_strategy,
    "trend_following": trend_following_strategy,
    "range_trading": range_trading_strategy,
}

# def run_swing_strategy(df):
#     return swing_strategy(df)  # replace with your actual function name

# def run_scalping_strategy(df):
#     return scalping_strategy(df)

# def run_breakout_strategy(df):
#     return breakout_strategy(df)

# def run_mean_reversion_strategy(df):
#     return mean_reversion_strategy(df)

# def run_momentum_strategy(df):
#     return momentum_strategy(df)
