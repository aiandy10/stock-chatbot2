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

st.set_page_config(page_title="Stock-Chatbot2", layout="wide")
st.title("📈 Stock-Chatbot2 — AI-Powered Trading Assistant")

# ----------------------------
# FETCH AVAILABLE STRATEGIES
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_strategies():
    try:
        response = requests.get(STRATEGIES_URL)
        if response.status_code == 200:
            return response.json().get("strategies", [])
        raise ValueError(f"Backend returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Could not fetch strategies from backend: {e}")
        return ["Swing Trading", "Scalping"]  # fallback

available_strategies = fetch_strategies()

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_nse_stocks():
    """Fetch NSE stocks from backend"""
    try:
        response = requests.get(f"{API_BASE}/stocks")
        if response.status_code == 200:
            return response.json()["stocks"]
    except Exception as e:
        st.error(f"Failed to fetch stocks: {e}")
    return []  # Return empty list if fetch fails

# Sidebar inputs
stocks = fetch_nse_stocks()
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    options=stocks,
    help="Start typing to search stocks"
)

# Remove .NS handling since it's handled in backend
stock = selected_stock  # Use selected stock directly

strategies = st.sidebar.multiselect(
    "Select Strategies",
    options=available_strategies,
    default=[available_strategies[0]] if available_strategies else [],
    help="Select one or more trading strategies to analyze"
)
length = st.sidebar.number_input("Data Length (days)", min_value=10, max_value=365, value=60)

# Add after strategy selection in sidebar
response_length = st.sidebar.select_slider(
    "Response Detail Level",
    options=["short", "medium", "long"],
    value="medium",
    help="Choose how detailed you want the analysis to be"
)

# ----------------------------
# RUN ANALYSIS
# ----------------------------
if st.sidebar.button("Run Analysis"):
    if not stock or not strategies:
        st.warning("Please enter a stock symbol and select at least one strategy.")
    else:
        with st.spinner("Fetching data and running strategies..."):
            payload = {"stock": stock, "strategies": strategies, "length": length, "response_length": response_length}  # Add this field
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
                    with tab1:
                        st.subheader("🧠 AI Summary")
                        st.markdown(data.get("groq_summary", "No summary available."))
                        st.markdown(f"**Final Signal:** {data.get('groq_signal', 'N/A')}")

                        st.subheader("📊 Strategy Breakdown")
                        for strategy, result in data.get("strategies", {}).items():
                            with st.expander(f"{strategy} Strategy", expanded=True):
                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    summary = result.get("summary", "No analysis available.")
                                    signal = result.get("signal", "N/A")
                                    
                                    # Format the summary with bullet points if needed
                                    if "•" in summary:
                                        points = summary.split("•")
                                        st.markdown("Analysis:")
                                        for point in points[1:]:  # Skip first empty split
                                            st.markdown(f"• {point.strip()}")
                                    else:
                                        st.markdown(summary)
                                    
                                    st.markdown(f"**Signal:** {signal}")
                                    
                                    # Show strategy-specific metrics if available
                                    metrics = result.get("metrics", {})
                                    if metrics:
                                        cols = st.columns(len(metrics))
                                        for col, (metric, value) in zip(cols, metrics.items()):
                                            col.metric(metric, value)

                    # Tab 2 — Technical Chart
                    with tab2:
                        # Prefer top-level price_data for robustness
                        price_data = data.get("price_data")

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
                        fundamentals_summary = data.get("fundamentals_summary", "")

                        if fundamentals_summary:
                            st.markdown("### Fundamentals Summary")
                            st.markdown(fundamentals_summary)

                        if fundamentals:
                            st.markdown("### Raw Fundamentals")
                            st.table(pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"]))
                        else:
                            st.info("No fundamentals data available.")

            except Exception as e:
                st.error(f"Request failed: {e}")
