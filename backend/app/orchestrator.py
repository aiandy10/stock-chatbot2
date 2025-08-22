import json
import pandas as pd
from app.llm import llm, RESPONSE_TEMPLATES
from app.services.stock_service import get_stock_info, get_stock_history
from app.services.fundamentals import get_fundamentals
from app.tools.strategy_tools import summarize_indicators, swing_strategy, scalping_strategy
from app.rag import get_context

def _format_fundamentals(f: dict) -> str:
    if not f:
        return "- No fundamentals available"
    return "\n".join(f"- {k}: {v}" for k, v in f.items() if v is not None)

def _format_indicators(ind: dict) -> str:
    if not ind:
        return "- No indicators available"
    # Flatten a bit for readability
    flat = {}
    for k, v in ind.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}.{sk}"] = sv
        else:
            flat[k] = v
    return "\n".join(f"- {k}: {v}" for k, v in flat.items() if v is not None)

def _strategy_output(strategy: str, stock: str, hist: pd.DataFrame) -> str:
    s = strategy.strip().lower()
    if s == "swing":
        return swing_strategy(stock, hist)
    if s == "scalping":
        return scalping_strategy(stock)
    return f"Strategy '{strategy}' not supported."

def run_analysis(stock: str, strategy: str, length: str) -> str:
    # Fetch data
    stock_info = get_stock_info(stock)
    history = get_stock_history(stock, period="6mo")
    fundamentals = get_fundamentals(stock)
    indicators = summarize_indicators(history)
    strategy_text = _strategy_output(strategy, stock, history)
    rag_context = get_context(stock)

    # Prompt
    template = RESPONSE_TEMPLATES.get(length, RESPONSE_TEMPLATES["medium"])
    recent_closes = ", ".join(f"{x:.2f}" for x in history["Close"].tail(5).tolist())

    prompt = f"""
You are a stock market analysis assistant.
Base your analysis ONLY on the provided data. If data is missing, say so. Do not make up numbers.

### Stock Information
- Name: {stock_info.get('longName')}
- Symbol: {stock_info.get('symbol', stock)}
- Current Price: {stock_info.get('currentPrice')}

### Fundamentals
{_format_fundamentals(fundamentals)}

### Technical Indicators
{_format_indicators(indicators)}

### Strategy Output ({strategy})
{strategy_text}

### Recent Closing Prices (last 5 days)
{recent_closes}

### External Market Context (RAG)
{rag_context}

Now, {template}
""".strip()

    # LLM call
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
