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
    strategies_text = "\n".join([f"- {name}: {summary}" for name, summary in strategy_summaries.items()])

    prompt = f"""
You are a senior equity research analyst. Your job is to provide an expert-level evaluation of {stock}.

Available information:
- Fundamentals:
{fundamentals_summary}

- Strategy Results:
{strategies_text}

- Contextual Information (retrieved from external data sources):
{context}

Instructions:
1. Write a **clear, structured analysis**. Cover:
   • Recent price action and trend  
   • Strengths and weaknesses from fundamentals  
   • Strategy alignment or conflicts (if strategies disagree, explain why)  
   • Short-term vs long-term outlook  
   • Key risks to watch  

2. Keep the tone professional, analytical, and actionable. Do not just list facts — interpret them.  

3. End with a **Final Signal**: one of [Buy, Sell, Hold].  
   - "Buy" if upside outweighs risks  
   - "Sell" if downside or overvaluation is likely  
   - "Hold" if evidence is mixed or neutral  

Output format:
Analysis:
<your detailed explanation, 3–5 paragraphs max>

Final Signal: <Buy/Sell/Hold>
"""

    resp = llm.invoke(prompt).content.strip()

    # Extract signal safely
    if "Buy" in resp:
        signal = "Buy"
    elif "Sell" in resp:
        signal = "Sell"
    else:
        signal = "Hold"

    return {
        "groq_summary": resp,
        "groq_signal": signal
    }
