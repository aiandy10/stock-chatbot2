# backend/app/llm.py
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.stock_service import get_stock_history, get_stock_info
from app.services.fundamentals import get_fundamentals
from app.tools.strategy_tools import swing_strategy  # add scalping, etc. later
from app.rag import get_context  # new RAG helper

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

# LangChain LLM wrapper for Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=500
)

# Output parser
parser = StrOutputParser()

# Response length templates
RESPONSE_TEMPLATES = {
    "short": "Give a concise 2-3 line analysis.",
    "medium": "Give a detailed analysis with key indicators.",
    "long": "Give a full detailed analysis, including historical data, fundamentals, and multiple strategies."
}

def get_groq_response(stock: str, strategy: str, length: str = "medium") -> str:
    """
    Orchestrates stock advice with Groq + LangChain + RAG.
    """
    # 1. Fetch data
    stock_info = get_stock_info(stock)
    fundamentals = get_fundamentals(stock)
    history = get_stock_history(stock, period="6mo")

    # 2. Strategy function
    strategy_map = {
        "swing": swing_strategy,
        # "scalping": scalping_strategy, etc.
    }
    strategy_fn = strategy_map.get(strategy)
    if not strategy_fn:
        return f"❌ Strategy '{strategy}' not supported."

    strategy_output = strategy_fn(stock)

    # 3. External context (RAG)
    rag_context = get_context(stock)

    # 4. Build prompt dynamically
    template = RESPONSE_TEMPLATES.get(length, RESPONSE_TEMPLATES["medium"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful stock market assistant. Always stay factual, never give financial advice."),
        ("user", 
f"""
Stock: {stock_info['longName']} ({stock})
Current Price: {stock_info['currentPrice']}

Fundamentals: {fundamentals}
Historical closing prices (last 10): {list(history['Close'].tail(10))}

Strategy Analysis: {strategy_output}
External Market Context (RAG): {rag_context}

Now, {template}
""")
    ])

    # 5. Run chain
    chain = prompt | llm | parser
    try:
        return chain.invoke({})
    except Exception as e:
        return f"❌ Error from Groq API: {str(e)}"
