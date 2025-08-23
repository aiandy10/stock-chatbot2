from typing import Dict, Any, List
import copy
import inspect
import re
import pandas as pd

from app.services.stock_service import get_stock_history, get_stock_info
from app.tools.strategy_tools import (
    swing_strategy,
    scalping_strategy,
    long_term_investment_strategy,
    position_trading_strategy,
    momentum_trading_strategy,
    mean_reversion_strategy,
    breakout_trading_strategy,
    trend_following_strategy,
    range_trading_strategy,
    summarize_indicators
)
from app.rag import get_context, get_groq_summary  # includes Groq summarizer

# --------------------------------------------------------------------------------------
# Strategy registry — strict naming convention
# --------------------------------------------------------------------------------------
STRATEGY_FUNCTIONS = {
    "swing": swing_strategy,
    "scalping": scalping_strategy,  # NOTE: takes (stock) only
    "long_term_investment": long_term_investment_strategy,
    "position_trading": position_trading_strategy,
    "momentum_trading": momentum_trading_strategy,
    "mean_reversion": mean_reversion_strategy,
    "breakout_trading": breakout_trading_strategy,
    "trend_following": trend_following_strategy,
    "range_trading": range_trading_strategy,
}

# Common aliases coming from UI/user-friendly labels -> strict keys above
STRATEGY_ALIASES = {
    "swing": "swing",
    "swing trading": "swing",

    "scalp": "scalping",
    "scalping": "scalping",
    "scalping strategy": "scalping",

    "long term investment": "long_term_investment",
    "long-term investment": "long_term_investment",
    "longterm": "long_term_investment",

    "position": "position_trading",
    "position trading": "position_trading",

    "momentum": "momentum_trading",
    "momentum trading": "momentum_trading",

    "mean reversion": "mean_reversion",
    "mean-reversion": "mean_reversion",
    "reversion": "mean_reversion",

    "breakout": "breakout_trading",
    "breakout trading": "breakout_trading",

    "trend": "trend_following",
    "trend following": "trend_following",

    "range": "range_trading",
    "range trading": "range_trading",
}

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _df_to_price_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert OHLCV DataFrame (Date index or first column) to records consumable by frontend.
    Ensures 'Date' is present and ISO stringified.
    """
    out = df.reset_index()
    if "Date" not in out.columns:
        out.rename(columns={out.columns[0]: "Date"}, inplace=True)
    # Normalize datetime -> string to avoid JSON serialization issues
    out["Date"] = pd.to_datetime(out["Date"]).astype(str)
    return out.to_dict(orient="records")


def _pack_indicators_with_price(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize indicators and attach canonical price_data so strategies can share it.
    """
    indicators = summarize_indicators(df)
    if not isinstance(indicators, dict):
        indicators = {"summary": indicators}
    indicators["price_data"] = _df_to_price_records(df)
    return indicators


def summarize_fundamentals(f: dict) -> str:
    """
    Format key fundamental ratios into a clean professional summary.
    """
    bullets = []
    if not isinstance(f, dict):
        return "No notable fundamentals."

    price = f.get("currentPrice")
    prev_close = f.get("previousClose")
    pe = f.get("trailingPE") or f.get("forwardPE")
    eps = f.get("trailingEps") or f.get("forwardEps")
    mcap = f.get("marketCap")
    div_yield = f.get("dividendYield")
    roe = f.get("returnOnEquity")
    profit_margin = f.get("profitMargins")

    if price and prev_close:
        try:
            change = ((price - prev_close) / prev_close) * 100
            bullets.append(f"Price: ₹{price:.2f} ({change:+.2f}% vs prev close ₹{prev_close:.2f})")
        except Exception:
            pass

    if pe:
        try:
            if pe < 15:
                bullets.append(f"P/E: {pe:.2f} — potentially undervalued")
            elif pe > 30:
                bullets.append(f"P/E: {pe:.2f} — priced for growth")
            else:
                bullets.append(f"P/E: {pe:.2f} — fairly valued")
        except Exception:
            pass

    if eps:
        try:
            bullets.append(f"EPS: {float(eps):.2f} — earnings strength indicator")
        except Exception:
            bullets.append(f"EPS: {eps} — earnings strength indicator")

    if mcap:
        try:
            size = "Large Cap" if mcap > 1e12 else "Mid/Small Cap"
            bullets.append(f"Market Cap: ₹{mcap/1e7:.2f} Cr — {size}")
        except Exception:
            pass

    if div_yield:
        try:
            bullets.append(f"Dividend Yield: {div_yield*100:.2f}% — income potential")
        except Exception:
            pass

    if roe:
        try:
            bullets.append(f"ROE: {roe*100:.2f}% — return efficiency")
        except Exception:
            pass

    if profit_margin:
        try:
            bullets.append(f"Profit Margin: {profit_margin*100:.2f}% — profitability snapshot")
        except Exception:
            pass

    # Return in structured professional format
    return "\n".join(bullets) or "No notable fundamentals."


def _normalize_strategy_name(name: str) -> str:
    """
    Normalize any incoming strategy label (UI-friendly) to the strict registry key.
    """
    key = (name or "").strip().lower()
    key = re.sub(r"\s+", " ", key)
    # direct hit first
    if key in STRATEGY_FUNCTIONS:
        return key
    # alias mapping
    return STRATEGY_ALIASES.get(key, key)


def _infer_signal_from_text(text: str) -> str:
    """
    Heuristic: infer Buy/Sell/Hold from a strategy's narrative.
    Prefer explicit mentions; fall back to bullish/bearish cues; else Neutral.
    """
    t = (text or "").lower()

    # Strong/explicit cues
    if re.search(r"\b(final )?signal:\s*buy\b", t):
        return "Buy"
    if re.search(r"\b(final )?signal:\s*sell\b", t):
        return "Sell"

    # Directional keywords
    buy_words = [
        "buy", "long bias", "prefer long", "bullish", "golden cross", "accumulation",
        "uptrend", "dip buy", "add on pullbacks", "momentum long"
    ]
    sell_words = [
        "sell", "short bias", "prefer short", "bearish", "death cross", "distribution",
        "downtrend", "reduce exposure", "avoid longs", "momentum short"
    ]

    if any(w in t for w in buy_words) and not any(w in t for w in sell_words):
        return "Buy"
    if any(w in t for w in sell_words) and not any(w in t for w in buy_words):
        return "Sell"

    # Mixed or unclear → Neutral/Hold
    return "Neutral"


def _call_strategy(fn, stock: str, df: pd.DataFrame) -> str:
    """
    Safely call a strategy function with the correct signature.
    Most strategies are (stock, df); scalping/day_trading are (stock).
    """
    # Try calling with (stock, df) first
    try:
        return fn(stock, df)  # standard path
    except TypeError:
        # Fallback: try (stock) only (e.g., scalping/day trading)
        return fn(stock)


# --------------------------------------------------------------------------------------
# Main Orchestrator
# --------------------------------------------------------------------------------------
def run_strategies(stock: str, strategies: List[str], length: int = 30) -> Dict[str, Any]:
    # 1) History
    df = get_stock_history(stock, period=f"{length}d")
    if df is None or df.empty:
        raise ValueError(f"No price history for {stock} ({length}d).")

    # 2) Fundamentals
    fundamentals = get_stock_info(stock) or {}
    fundamentals_summary = summarize_fundamentals(fundamentals)

    # 3) RAG context (can be empty if DB cold)
    rag_context = get_context(stock) or ""

    # 4) Shared indicators + price data (canonical source for frontend chart)
    shared_indicators = _pack_indicators_with_price(df)
    shared_price_data = shared_indicators["price_data"]

    # 5) Execute strategies
    results: Dict[str, Any] = {}
    for strat_in in strategies:
        # Keep the *display* key as the user provided (nice in UI),
        # but resolve to our strict key for function lookup.
        display_key = (strat_in or "").strip()
        lookup_key = _normalize_strategy_name(display_key)

        fn = STRATEGY_FUNCTIONS.get(lookup_key)
        if not fn:
            results[display_key] = {"error": f"Strategy '{display_key}' not implemented"}
            continue

        try:
            summary_text: str = _call_strategy(fn, stock, df)
            summary_text = (summary_text or "").strip()
            # infer a per-strategy signal so non-swing strategies don't look generic
            inferred_signal = _infer_signal_from_text(summary_text)

            # Use a deepcopy so per-strategy consumers can mutate indicators safely if needed
            results[display_key] = {
                "summary": summary_text,
                "signal": inferred_signal,
                "indicators": copy.deepcopy(shared_indicators),
            }
        except Exception as e:
            results[display_key] = {"error": f"{type(e).__name__}: {e}"}

    # 6) Collate summaries for LLM
    strategy_summaries = {k: v.get("summary", "") for k, v in results.items() if "summary" in v}

    # 7) Groq analysis + final signal
    groq_output = get_groq_summary(stock, fundamentals_summary, strategy_summaries, rag_context)

    # 8) Response (now includes top-level price_data for robust charting)
    return {
        "stock": stock,
        "price_data": shared_price_data,                    # ✅ frontend should use this directly
        "fundamentals": fundamentals,
        "fundamentals_summary": fundamentals_summary,
        "strategies": results,
        "rag_context": rag_context,
        "groq_summary": groq_output.get("groq_summary", "").strip(),
        "groq_signal": groq_output.get("groq_signal", "Neutral"),
    }
