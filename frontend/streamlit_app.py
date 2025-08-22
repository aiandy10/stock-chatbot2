import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# CONFIG
# ----------------------------
API_BASE = "http://localhost:8000"  # Change if backend is hosted elsewhere
ANALYZE_URL = f"{API_BASE}/analyze"
STRATEGIES_URL = f"{API_BASE}/strategies"

st.set_page_config(page_title="Stock‑Chatbot2", layout="wide")
st.title("📈 Stock‑Chatbot2 — AI‑Powered Trading Assistant")

# ----------------------------
# FETCH AVAILABLE STRATEGIES
# ----------------------------
try:
    available_strategies = requests.get(STRATEGIES_URL).json()
    if not isinstance(available_strategies, list):
        raise ValueError("Invalid strategies format from backend")
except Exception as e:
    st.error(f"⚠️ Could not fetch strategies from backend: {e}")
    available_strategies = ["Swing", "Scalping"]  # fallback

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
stock = st.sidebar.text_input("Stock Symbol", value="TCS.NS")
strategies = st.sidebar.multiselect(
    "Select Strategies",
    available_strategies,
    default=[available_strategies[0]] if available_strategies else []
)
length = st.sidebar.number_input("Data Length (days)", min_value=10, max_value=365, value=60)

# ----------------------------
# RUN ANALYSIS
# ----------------------------
if st.sidebar.button("Run Analysis"):
    if not stock or not strategies:
        st.warning("Please enter a stock symbol and select at least one strategy.")
    else:
        with st.spinner("Fetching data and running strategies..."):
            payload = {"stock": stock, "strategies": strategies, "length": length}
            try:
                response = requests.post(ANALYZE_URL, json=payload)
                if response.status_code != 200:
                    st.error(f"Error {response.status_code}: {response.text}")
                else:
                    data = response.json()

                    # ----------------------------
                    # TABS
                    # ----------------------------
                    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Technical Chart", "📑 Fundamentals"])

                    # Tab 1 — Analysis
                    # Tab 1 — Analysis
                    with tab1:
                        st.subheader("🧠 AI Summary")
                        st.markdown(data.get("groq_summary", "No summary available."))
                        st.markdown(f"**Final Signal:** {data.get('groq_signal', 'N/A')}")
                    
                        st.subheader("📊 Strategy Breakdown")
                        for strat, result in data.get("strategies", {}).items():
                            st.markdown(f"### {strat} Strategy")
                            st.markdown(result.get("summary", "No summary available."))
                            st.markdown(f"**Signal:** {result.get('signal', 'N/A')}")

                    # Tab 2 — Technical Chart
                    with tab2:
                        # Try to get price data from the first strategy's indicators
                        first_strat = next(iter(data.get("strategies", {}).values()), {})
                        price_data = first_strat.get("indicators", {}).get("price_data")

                        if price_data:
                            df = pd.DataFrame(price_data)
                            if "Date" in df.columns:
                                df["Date"] = pd.to_datetime(df["Date"])
                            fig = go.Figure(data=[go.Candlestick(
                                x=df["Date"],
                                open=df["Open"],
                                high=df["High"],
                                low=df["Low"],
                                close=df["Close"]
                            )])
                            fig.update_layout(title=f"{stock} Price Chart", xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No price data available for chart.")

                    # Tab 3 — Fundamentals
                    with tab3:
                        fundamentals = data.get("fundamentals", {})
                        if fundamentals:
                            st.table(pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"]))
                        else:
                            st.info("No fundamentals data available.")

            except Exception as e:
                st.error(f"Request failed: {e}")
