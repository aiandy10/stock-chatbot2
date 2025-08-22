# import yfinance as yf

# def get_stock_info(ticker: str):
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info

#         return {
#             "symbol": ticker.upper(),
#             "longName": info.get("longName"),
#             "currentPrice": info.get("currentPrice"),
#             "previousClose": info.get("previousClose"),
#             "open": info.get("open"),
#             "dayHigh": info.get("dayHigh"),
#             "dayLow": info.get("dayLow"),
#             "marketCap": info.get("marketCap"),
#             "volume": info.get("volume")
#         }
#     except Exception as e:
#         return {"error": str(e)}

# services/stock_service.py

import yfinance as yf

def get_stock_history(symbol: str, period="6mo"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    if hist is None or hist.empty:
        raise ValueError(f"No history returned for {symbol}")
    return hist  # Date index; columns: Open, High, Low, Close, Volume

def get_stock_info(symbol: str):
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        "symbol": symbol,
        "longName": info.get("longName"),
        "currentPrice": info.get("currentPrice"),
        "previousClose": info.get("previousClose"),
        "open": info.get("open"),
        "dayHigh": info.get("dayHigh"),
        "dayLow": info.get("dayLow"),
        "marketCap": info.get("marketCap"),
        "volume": info.get("volume"),
    }

# --- Wrappers to match orchestrator.py expectations ---

def fetch_price_data(symbol: str, length: int = 30):
    """
    Wrapper for get_stock_history to match orchestrator's expected name.
    Converts length (days) into yfinance period format.
    """
    return get_stock_history(symbol, period=f"{length}d")

def fetch_fundamentals(symbol: str):
    """
    Wrapper for get_stock_info to match orchestrator's expected name.
    """
    return get_stock_info(symbol)
