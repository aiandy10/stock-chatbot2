# import streamlit as st
# import requests

# # Backend API URL
# API_URL = "http://127.0.0.1:8000"

# st.set_page_config(page_title="Stock Chatbot 2", layout="centered")

# st.title("🚀 Stock Chatbot 2")
# st.write("Select a stock, strategy, and response length to get trading insights.")

# # --- Dropdowns ---
# # Example NSE stock list; you can expand this later
# stocks = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
# strategies = ["swing", "scalping"]  # Add more strategies as implemented
# lengths = ["short", "medium", "long"]

# selected_stock = st.selectbox("Choose Stock", stocks)
# selected_strategy = st.selectbox("Choose Strategy", strategies)
# selected_length = st.selectbox("Choose Response Length", lengths)

# # --- Button ---
# if st.button("Get Analysis"):
#     with st.spinner("Fetching insights..."):
#         try:
#             # Build a combined query string
#             question = (
#                 f"Analyze {selected_stock} "
#                 f"using {selected_strategy} strategy "
#                 f"with a {selected_length} response "
#                 f"using real historical stock data."
#             )

#             payload = {"query": question}

#             response = requests.post(f"{API_URL}/chat", json=payload)
#             data = response.json()
#             st.subheader("💡 Stock Analysis")
#             st.write(data.get("answer", "No response received."))
#         except Exception as e:
#             st.error(f"Error connecting to backend: {str(e)}")


# frontend/streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Stock-Chatbot2", layout="centered")

st.title("📈 Stock-Chatbot2")

# Dropdowns
stock = st.text_input("Enter NSE Stock Symbol", value="TCS.NS")
strategy = st.selectbox("Strategy", ["RSI", "MACD", "Ichimoku", "MA Crossover"])
length = st.selectbox("Response Length", ["short", "medium", "long"])

if st.button("Analyze"):
    if stock and strategy and length:
        payload = {"stock": stock, "strategy": strategy, "length": length}
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json=payload)
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(f"**Analysis for {stock} ({strategy} | {length}):**")
                st.write(answer)
            else:
                st.error(f"Error connecting to backend: {response.status_code}")
        except Exception as e:
            st.error(f"Exception: {e}")


