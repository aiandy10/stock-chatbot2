from typing import Dict, Any
from langgraph.graph import StateGraph, END

# Import your existing modules
from app.services.stock_service import get_stock_info, get_stock_history
from app.services.fundamentals import get_fundamentals
from app.tools.strategy_tools import swing_strategy, scalping_strategy
from app.rag import get_context
from app.llm import llm, RESPONSE_TEMPLATES


# Define the "state" object that flows between nodes
class BotState(Dict[str, Any]): 
    pass


# ---- Nodes ----
def fetch_stock_info(state: BotState) -> BotState:
    stock = state["stock"]
    state["stock_info"] = get_stock_info(stock)
    return state


def fetch_fundamentals(state: BotState) -> BotState:
    stock = state["stock"]
    state["fundamentals"] = get_fundamentals(stock)
    return state


def fetch_history(state: BotState) -> BotState:
    stock = state["stock"]
    state["history"] = get_stock_history(stock, period="6mo")
    return state


def apply_strategy(state: BotState) -> BotState:
    strategy = state.get("strategy", "swing")

    strategies = {
        "swing": swing_strategy,
        "scalping": scalping_strategy,
    }

    fn = strategies.get(strategy)
    if not fn:
        state["strategy_output"] = f"Strategy {strategy} not supported"
    else:
        state["strategy_output"] = fn(state["stock"])

    return state


def rag_context(state: BotState) -> BotState:
    stock = state["stock"]
    state["rag_context"] = get_context(stock)
    return state


def llm_response(state: BotState) -> BotState:
    template = RESPONSE_TEMPLATES.get(state["length"], RESPONSE_TEMPLATES["medium"])

    prompt = f"""
Stock: {state['stock_info'].get('longName')} ({state['stock']})
Price: {state['stock_info'].get('currentPrice')}
Fundamentals: {state.get('fundamentals')}
Recent Close: {list(state['history']['Close'].tail(5))}
Strategy Output: {state.get('strategy_output')}
Context: {state.get('rag_context')}

Now, {template}
"""

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# ---- Graph Definition ----
def build_graph():
    workflow = StateGraph(BotState)

    # Add nodes
    workflow.add_node("stock_info", fetch_stock_info)
    workflow.add_node("fundamentals", fetch_fundamentals)
    workflow.add_node("history", fetch_history)
    workflow.add_node("strategy", apply_strategy)
    workflow.add_node("rag", rag_context)
    workflow.add_node("llm", llm_response)

    # Set entry point
    workflow.set_entry_point("stock_info")

    # Execution order
    workflow.add_edge("stock_info", "fundamentals")
    workflow.add_edge("fundamentals", "history")
    workflow.add_edge("history", "strategy")
    workflow.add_edge("strategy", "rag")
    workflow.add_edge("rag", "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()
