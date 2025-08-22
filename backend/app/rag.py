# # app/rag.py
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# load_dotenv()


# # Load Groq API key from env
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Embeddings
# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Persistent DB
# DB_DIR = "db/chroma"

# # Text splitter
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# # Groq LLM
# llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# def load_docs(raw_texts: list[str]):
#     docs = [Document(page_content=t) for t in raw_texts]
#     split_docs = splitter.split_documents(docs)
#     vectorstore = Chroma.from_documents(split_docs, embed_model, persist_directory=DB_DIR)
#     vectorstore.persist()
#     return vectorstore

# def get_retriever():
#     return Chroma(persist_directory=DB_DIR, embedding_function=embed_model).as_retriever(
#         search_type="similarity", search_kwargs={"k": 3}
#     )

# def answer_question(query: str) -> str:
#     retriever = get_retriever()
#     docs = retriever.get_relevant_documents(query)

#     context = "\n".join([d.page_content for d in docs])
#     prompt = f"""You are a stock market assistant. 
# Use the following context to answer the question:

# Context:
# {context}

# Question: {query}
# Answer:"""

#     response = llm.invoke(prompt)
#     return response.content if hasattr(response, "content") else str(response)


# backend/app/rag.py
# backend/app/rag.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
import yfinance as yf

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent Chroma DB
DB_DIR = "db/chroma"

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

def fetch_stock_data(symbol: str, period="1y"):
    """Fetch historical stock data using yfinance"""
    df = yf.Ticker(symbol).history(period=period)
    df_str = df.to_csv()
    return df_str

def load_docs(stock_symbol: str):
    """Ingest stock historical data into Chroma"""
    raw_texts = [fetch_stock_data(stock_symbol)]
    docs = [Document(page_content=t) for t in raw_texts]
    split_docs = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(split_docs, embed_model, persist_directory=DB_DIR)
    vectorstore.persist()
    return vectorstore

def get_retriever():
    return Chroma(persist_directory=DB_DIR, embedding_function=embed_model).as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

def answer_question(stock: str, strategy: str, length: str) -> str:
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(stock)

    context = "\n".join([d.page_content for d in docs])
    prompt = f"""You are a stock market assistant. Provide a {length} analysis using the {strategy} strategy for {stock}.
Use the following context (historical/fundamental stock data):

{context}

Answer:"""

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
