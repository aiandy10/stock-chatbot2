# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import answer_question, load_docs

# Init app
app = FastAPI(
    title="Stock AI Assistant",
    description="AI-powered stock market assistant with scalping, swing trading, and RAG",
    version="1.0.0",
)

# Enable CORS for frontend/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    query: str

class IngestRequest(BaseModel):
    texts: list[str]

@app.get("/")
async def root():
    return {"message": "✅ Stock AI Assistant is running!"}


@app.post("/chat")
async def chat(query: Query):
    """
    Main AI chat endpoint.
    Takes a query string, retrieves context with RAG, and returns Groq LLM answer.
    """
    response = answer_question(query.query)
    return {"answer": response}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """
    Ingest raw texts into Chroma DB for RAG.
    Example body:
    {
      "texts": ["RSI is good for scalping when tuned between 1-min and 5-min charts."]
    }
    """
    vectorstore = load_docs(req.texts)
    return {"status": "✅ Documents ingested", "docs": len(req.texts)}
