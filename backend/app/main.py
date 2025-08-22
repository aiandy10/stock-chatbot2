# # backend/app/main.py

# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from app.rag import answer_question, load_docs

# # Init app
# app = FastAPI(
#     title="Stock AI Assistant",
#     description="AI-powered stock market assistant with scalping, swing trading, and RAG",
#     version="1.0.0",
# )

# # Enable CORS for frontend/dev
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # in prod, restrict this
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request models
# class QueryBody(BaseModel):
#     query: str

# class IngestRequest(BaseModel):
#     texts: list[str]

# @app.get("/")
# async def root():
#     return {"message": "✅ Stock AI Assistant is running!"}


# # ✅ GET version for Streamlit frontend
# @app.get("/chat")
# async def chat_get(
#     stock: str = Query(...),
#     strategy: str = Query(...),
#     length: str = Query(...)
# ):
#     query = f"Give a {length} {strategy} trading analysis for {stock}."
#     response = answer_question(query)
#     return {"answer": response}


# # ✅ POST version for flexibility
# @app.post("/chat")
# async def chat_post(query: QueryBody):
#     response = answer_question(query.query)
#     return {"answer": response}


# @app.post("/ingest")
# async def ingest(req: IngestRequest):
#     """
#     Ingest raw texts into Chroma DB for RAG.
#     Example body:
#     {
#       "texts": ["RSI is good for scalping when tuned between 1-min and 5-min charts."]
#     }
#     """
#     load_docs(req.texts)
#     return {"status": "✅ Documents ingested", "docs": len(req.texts)}

# backend/app/main.py
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import answer_question, load_docs

app = FastAPI(
    title="Stock AI Assistant",
    description="AI-powered stock market assistant with RAG",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatQuery(BaseModel):
    stock: str
    strategy: str
    length: str

@app.get("/")
async def root():
    return {"message": "✅ Stock AI Assistant running"}

@app.post("/chat")
async def chat(query: ChatQuery):
    """Main chat endpoint"""
    response = answer_question(query.stock, query.strategy, query.length)
    return {"answer": response}

@app.post("/ingest/{stock_symbol}")
async def ingest(stock_symbol: str):
    """Ingest historical data for a stock"""
    load_docs(stock_symbol)
    return {"status": f"✅ {stock_symbol} data ingested"}
