# import streamlit as st
# import requests
# import pandas as pd
# import plotly.graph_objects as go
# from io import StringIO

# # --- Backend API URL ---
# # If running in Codespaces, replace with your forwarded backend URL
# API_URL = "http://127.0.0.1:8000"

# st.set_page_config(page_title="Stock-Chatbot2", layout="wide")
# st.title("📈 Stock-Chatbot2 — Data‑Driven Stock Analysis")

# # --- Sidebar Inputs ---
# st.sidebar.header("Analysis Settings")
# stock = st.sidebar.text_input("Enter NSE Stock Symbol", value="TCS.NS")
# strategy = st.sidebar.selectbox("Strategy", ["swing", "scalping"])
# length = st.sidebar.selectbox("Response Length", ["short", "medium", "long"])

# if st.sidebar.button("Run Analysis"):
#     if stock and strategy and length:
#         payload = {
#             "stock": stock,
#             "strategy": strategy,
#             "length": length
#         }
#         try:
#             with st.spinner("Fetching analysis..."):
#                 # --- Call backend /chat ---
#                 response = requests.post(f"{API_URL}/chat", json=payload)
#                 if response.status_code == 200:
#                     answer = response.json().get("answer", "No answer returned.")

#                     # --- Tabs ---
#                     tab1, tab2, tab3 = st.tabs(["💡 Analysis", "📊 Technical Chart", "📑 Fundamentals"])

#                     # Tab 1: LLM Analysis
#                     with tab1:
#                         st.markdown(f"### Analysis for **{stock}** ({strategy} | {length})")
#                         st.write(answer)

#                     # Tab 2: Technical Chart
#                     with tab2:
#                         try:
#                             hist_url = f"https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1=0&period2=9999999999&interval=1d&events=history"
#                             hist_resp = requests.get(hist_url)
#                             if hist_resp.status_code == 200:
#                                 df = pd.read_csv(StringIO(hist_resp.text))
#                                 fig = go.Figure(data=[go.Candlestick(
#                                     x=df['Date'],
#                                     open=df['Open'],
#                                     high=df['High'],
#                                     low=df['Low'],
#                                     close=df['Close'],
#                                     name="Price"
#                                 )])
#                                 fig.update_layout(title=f"{stock} Price Chart", xaxis_rangeslider_visible=False)
#                                 st.plotly_chart(fig, use_container_width=True)
#                             else:
#                                 st.warning("Could not fetch chart data.")
#                         except Exception as e:
#                             st.error(f"Chart error: {e}")

#                     # Tab 3: Fundamentals
#                     with tab3:
#                         try:
#                             fund_resp = requests.get(f"{API_URL}/fundamentals/{stock}")
#                             if fund_resp.status_code == 200:
#                                 fundamentals = fund_resp.json()
#                                 st.table(pd.DataFrame(list(fundamentals.items()), columns=["Metric", "Value"]))
#                             else:
#                                 st.warning("Could not fetch fundamentals.")
#                         except Exception as e:
#                             st.error(f"Fundamentals error: {e}")

#                 else:
#                     st.error(f"Backend error: {response.status_code}")
#         except Exception as e:
#             st.error(f"Exception: {e}")

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"  # use your Codespaces forwarded URL if needed

st.set_page_config(page_title="Stock-Chatbot2", layout="centered")
st.title("📈 Stock-Chatbot2 — Grounded Analysis")

stock = st.text_input("NSE Symbol", value="TCS.NS")
strategy = st.selectbox("Strategy", ["swing", "scalping"])
length = st.selectbox("Response Length", ["short", "medium", "long"])

if st.button("Analyze"):
    payload = {"stock": stock, "strategy": strategy, "length": length}
    try:
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
        st.json(resp.json())
    except Exception as e:
        st.error(str(e))

