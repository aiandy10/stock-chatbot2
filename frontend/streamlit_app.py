import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stock Chatbot 2", layout="centered")

st.title("🚀 Stock Chatbot 2")
st.write("Select a stock, strategy, and response length to get trading insights.")

# --- Dropdowns ---
# Example NSE stock list; you can expand this later
stocks = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
strategies = ["swing", "scalping"]  # Add more strategies as implemented
lengths = ["short", "medium", "long"]

selected_stock = st.selectbox("Choose Stock", stocks)
selected_strategy = st.selectbox("Choose Strategy", strategies)
selected_length = st.selectbox("Choose Response Length", lengths)

# --- Button ---
if st.button("Get Analysis"):
    with st.spinner("Fetching insights..."):
        try:
            response = requests.get(
                f"{API_URL}/chat",
                params={
                    "stock": selected_stock,
                    "strategy": selected_strategy,
                    "length": selected_length
                }
            )
            data = response.json()
            st.subheader("💡 Stock Analysis")
            st.write(data.get("answer", "No response received."))
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
