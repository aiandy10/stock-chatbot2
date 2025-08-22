import os
import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from typing import Dict


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_DIR = "db/chroma"
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

def fetch_stock_data(symbol: str, period="1y"):
    df = yf.Ticker(symbol).history(period=period)
    return df.to_csv()

def load_docs(stock_symbol: str):
    raw_texts = [fetch_stock_data(stock_symbol)]
    docs = [Document(page_content=t) for t in raw_texts]
    split_docs = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(split_docs, embed_model, persist_directory=DB_DIR)
    vectorstore.persist()
    return vectorstore

def get_retriever():
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embed_model
    ).as_retriever(search_type="similarity", search_kwargs={"k": 3})

def get_context(stock: str) -> str:
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(stock)
    return "\n".join(d.page_content for d in docs)


def get_groq_summary(stock: str, fundamentals_summary: str, strategy_summaries: Dict[str, str], context: str) -> Dict[str, str]:
    prompt = f"""
You are a financial analyst. Analyze the following data for {stock} and produce a concise summary and a Buy/Sell/Hold signal.

Fundamentals Summary:
{fundamentals_summary}

Strategy Summaries:
{chr(10).join(f"{k}: {v}" for k, v in strategy_summaries.items())}

Contextual Info:
{context}

Respond with:
- A short summary in bullet points
- A final signal: Buy, Sell, or Hold
"""
    response = llm.invoke(prompt)
    return {
        "groq_summary": response.content.strip(),
        "groq_signal": ("Buy" if "Buy" in response.content else
                        "Sell" if "Sell" in response.content else
                        "Hold")
    }