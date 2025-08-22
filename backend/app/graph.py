from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.services.stock_service import get_stock_info, get_stock_history
from app.services.fundamentals import get_fundamentals
from app.tools.strategy_tools import summarize_indicators, swing_strategy, scalping_strategy
from app.rag import get_context
from app.llm import llm, RESPONSE_TEMPLATES

class BotState(Dict[str, Any]): 
    pass

def fetch_stock_info(state: BotState) -> BotState:
    state["stock_info"] = get_stock_info(state["stock"])
    return state

def fetch_fundamentals(state: BotState) -> BotState:
    state["fundamentals"] = get_fundamentals(state["stock"])
    return state

def fetch_history(state: BotState) -> BotState:
    state["history"] = get_stock_history(state["stock"], period="6mo")
    return state

def apply_strategy(state: BotState) -> BotState:
    df = state["history"]
    state["indicators"] = summarize_indicators(df)

    strategies = {
        "swing": lambda s: swing_strategy(s, df),
        "scalping": scalping_strategy
    }
    fn = strategies.get(state["strategy"].lower())
    state["strategy_output"] = fn(state["stock"]) if fn else f"Strategy {state['strategy']} not supported"
    return state

def rag_context(state: BotState) -> BotState:
    state["rag_context"] = get_context(state["stock"])
    return state

def llm_response(state: BotState) -> BotState:
    template = RESPONSE_TEMPLATES.get(state["length"], RESPONSE_TEMPLATES["medium"])

    fundamentals_str = "\n".join(
        f"- {k}: {v}" for k, v in state.get("fundamentals", {}).items() if v is not None
    )
    indicators_str = "\n".join(
        f"- {k}: {v}" for k, v in state.get("indicators", {}).items() if v is not None
    )
    recent_closes = ", ".join(f"{p:.2f}" for p in state["history"]["Close"].tail(5))

    prompt = f"""
You are a stock market analysis assistant.
Base your analysis ONLY on the provided data.
If data is missing, say so. Do not make up numbers.

### Stock Information
- Name: {state['stock_info'].get('longName')}
- Symbol: {state['stock']}
- Current Price: {state['stock_info'].get('currentPrice')}

### Fundamentals
{fundamentals_str}

### Technical Indicators
{indicators_str}

### Strategy Output ({state['strategy']}):
{state.get('strategy_output')}

### Recent Closing Prices (last 5 days)
{recent_closes}

### External Market Context (RAG)
{state.get('rag_context')}

Now, {template}
"""

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state

def build_graph():
    workflow = StateGraph(BotState)
    workflow.add_node("stock_info", fetch_stock_info)
    workflow.add_node("fundamentals", fetch_fundamentals)
    workflow.add_node("history", fetch_history)
    workflow.add_node("strategy", apply_strategy)
    workflow.add_node("rag", rag_context)
    workflow.add_node("llm", llm_response)

    workflow.set_entry_point("stock_info")
    workflow.add_edge("stock_info", "fundamentals")
    workflow.add_edge("fundamentals", "history")
    workflow.add_edge("history", "strategy")
    workflow.add_edge("strategy", "rag")
    workflow.add_edge("rag", "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()
