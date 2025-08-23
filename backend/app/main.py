# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from app.graph import build_graph
# from app.services.fundamentals import get_fundamentals

# # --- FastAPI App ---
# app = FastAPI(
#     title="Stock AI Assistant",
#     description="AI-powered stock market assistant with RAG + Technical/Fundamental Analysis",
#     version="1.0.0",
# )

# # --- CORS ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, restrict this
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Request Models ---
# class ChatQuery(BaseModel):
#     stock: str
#     strategy: str
#     length: str
#     response_length: str = "medium"  # Add this field with default

# # --- Routes ---
# @app.get("/")
# async def root():
#     return {"message": "✅ Stock AI Assistant running"}

# @app.post("/chat")
# async def chat(query: ChatQuery):
#     """
#     Main chat endpoint using LangGraph pipeline.
#     Runs fundamentals, indicators, strategy output, and RAG context
#     through a structured prompt to Groq.
#     """
#     workflow = build_graph()
#     state = workflow.invoke({
#         "stock": query.stock,
#         "strategy": query.strategy,
#         "length": query.length
#     })
#     return {"answer": state["answer"]}

# @app.get("/fundamentals/{stock_symbol}")
# async def fundamentals(stock_symbol: str):
#     """
#     Return fundamentals for a given stock symbol.
#     Used by the frontend's Fundamentals tab.
#     """
#     return get_fundamentals(stock_symbol)

# # @app.post("/chat")
# # async def chat(query: ChatQuery):
# #     print("DEBUG incoming query:", query.dict())  # See exactly what came in
# #     workflow = build_graph()
# #     try:
# #         state = workflow.invoke({
# #             "stock": query.stock,
# #             "strategy": query.strategy,
# #             "length": query.length
# #         })
# #         return {"answer": state["answer"]}
# #     except Exception as e:
# #         import traceback
# #         traceback.print_exc()  # Print full error to terminal
# #         return {"error": str(e)}

# @app.post("/chat")
# async def chat(query: ChatQuery):
#     print("DEBUG incoming query:", query.dict())  # <--- Add this
#     workflow = build_graph()
#     state = workflow.invoke({
#         "stock": query.stock,
#         "strategy": query.strategy,
#         "length": query.length
#     })
#     return {"answer": state["answer"]}

# backend/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.orchestrator import run_strategies
from app.services.nse_stocks import fetch_nse_stocks


# Initialize FastAPI app
app = FastAPI(
    title="Stock‑Chatbot2 API",
    description="Backend API for Stock‑Chatbot2 — runs strategies and returns data‑grounded analysis",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class StrategyRequest(BaseModel):
    stock: str
    strategies: list
    length: int = 30

# Health check route
@app.get("/")
async def root():
    return {"status": "ok", "message": "Stock‑Chatbot2 backend is running"}

# Main analysis route
@app.post("/analyze")
async def analyze(request: StrategyRequest):
    """
    Accepts stock symbol, list of strategies, and optional length.
    Returns fundamentals, strategy outputs, and RAG context.
    """
    result = run_strategies(
        stock=request.stock,
        strategies=request.strategies,
        length=request.length
    )
    return result

@app.get("/strategies")
async def get_strategies():
    """Get available trading strategies"""
    strategies = [
        "Swing Trading",
        "Scalping",
        "Long-term Investment",
        "Position Trading",
        "Momentum Trading",
        "Mean Reversion",
        "Breakout Trading",
        "Trend Following",
        "Range Trading",
        "Day Trading",
        "Value Investing",
        "Growth Investing",
        "Dividend Growth",
        "Options Trading"
    ]
    return {"strategies": strategies}

@app.get("/stocks")
async def get_stocks():
    """Get list of NSE stocks for autocomplete"""
    try:
        stocks = fetch_nse_stocks()
        return {"stocks": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

