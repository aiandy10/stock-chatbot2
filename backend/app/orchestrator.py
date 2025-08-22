from typing import Dict, Any, List
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
from app.rag import get_context  # corrected name

# --------------------------------------------------------------------------------------
# Strategy registry — strict naming convention
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _df_to_price_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Normalize OHLCV data for frontend charts."""
    out = df.reset_index()
    if "Date" not in out.columns:
        out.rename(columns={out.columns[0]: "Date"}, inplace=True)
    out["Date"] = pd.to_datetime(out["Date"]).astype(str)
    return out.to_dict(orient="records")

def _pack_indicators_with_price(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute full indicator summary and embed price_data for chart."""
    indicators = summarize_indicators(df)
    if not isinstance(indicators, dict):
        indicators = {"summary": indicators}
    indicators["price_data"] = _df_to_price_records(df)
    return indicators

def summarize_fundamentals(f: dict) -> str:
    """Turn raw fundamentals dict into a concise, number‑driven narrative."""
    bullets = []
    price = f.get("currentPrice")
    prev_close = f.get("previousClose")
    pe = f.get("trailingPE") or f.get("forwardPE")
    eps = f.get("trailingEps") or f.get("forwardEps")
    mcap = f.get("marketCap")
    div_yield = f.get("dividendYield")
    roe = f.get("returnOnEquity")
    profit_margin = f.get("profitMargins")

    if price and prev_close:
        change = ((price - prev_close) / prev_close) * 100
        bullets.append(f"Price ₹{price:.2f}, {change:+.2f}% vs prev close ₹{prev_close:.2f}.")

    if pe:
        if pe < 15:
            bullets.append(f"PE {pe:.2f} — potentially undervalued.")
        elif pe > 30:
            bullets.append(f"PE {pe:.2f} — priced for growth.")
        else:
            bullets.append(f"PE {pe:.2f} — fairly valued.")

    if eps:
        bullets.append(f"EPS {eps:.2f} — earnings strength indicator.")

    if mcap:
        size = "large" if mcap > 1e12 else "mid/small"
        bullets.append(f"Market cap ₹{mcap/1e7:.2f} Cr — {size} cap profile.")

    if div_yield:
        bullets.append(f"Dividend yield {div_yield*100:.2f}% — income potential.")

    if roe:
        bullets.append(f"ROE {roe*100:.2f}% — return efficiency.")

    if profit_margin:
        bullets.append(f"Profit margin {profit_margin*100:.2f}% — profitability snapshot.")

    return "\n".join(f"• {b}" for b in bullets) or "No notable fundamentals."

# --------------------------------------------------------------------------------------
# Main Orchestrator
# --------------------------------------------------------------------------------------
def run_strategies(stock: str, strategies: List[str], length: int = 30) -> Dict[str, Any]:
    """
    Fetches data, runs selected strategies (stock, df),
    and returns structured + narrative results for frontend & LLM.
    """
    # 1) Price history
    df = get_stock_history(stock, period=f"{length}d")
    if df is None or df.empty:
        raise ValueError(f"No price history for {stock} ({length}d).")

    # 2) Fundamentals
    fundamentals = get_stock_info(stock)
    fundamentals_summary = summarize_fundamentals(fundamentals)

    # 3) Context (news, filings, historical notes)
    rag_context = get_context(stock)

    # 4) Indicators snapshot (shared for all strategies)
    shared_indicators = _pack_indicators_with_price(df)

    # 5) Run strategies
    results: Dict[str, Any] = {}
    for strat_in in strategies:
        key = (strat_in or "").strip().lower()
        fn = STRATEGY_FUNCTIONS.get(key)
        if not fn:
            results[strat_in] = {"error": f"Strategy '{strat_in}' not implemented"}
            continue

        try:
            summary_text: str = fn(stock, df)  # rich bullet‑point analysis
            results[strat_in] = {
                "summary": summary_text,
                "indicators": shared_indicators,
            }
        except Exception as e:
            results[strat_in] = {"error": f"{type(e).__name__}: {e}"}

    # 6) Return payload
    return {
        "stock": stock,
        "fundamentals": fundamentals,               # full dict for LLM grounding
        "fundamentals_summary": fundamentals_summary, # human‑readable for Analysis tab
        "strategies": results,
        "rag_context": rag_context
    }
