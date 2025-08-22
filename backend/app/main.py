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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.orchestrator import run_analysis
from app.services.fundamentals import get_fundamentals

app = FastAPI(
    title="Stock AI Assistant",
    description="Grounded stock analysis with fundamentals, indicators, strategy, and RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    stock: str
    strategy: str
    length: str

@app.get("/")
async def root():
    return {"message": "✅ Stock AI Assistant running"}

@app.post("/chat")
async def chat(query: ChatQuery):
    # Hard guard to prevent empty state
    if not query.stock or not query.strategy or not query.length:
        raise HTTPException(status_code=400, detail="Missing fields: stock, strategy, length")
    try:
        answer = run_analysis(stock=query.stock, strategy=query.strategy, length=query.length)
        return {"answer": answer}
    except Exception as e:
        # Surface the error to help you debug
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/fundamentals/{stock_symbol}")
async def fundamentals(stock_symbol: str):
    return get_fundamentals(stock_symbol)

