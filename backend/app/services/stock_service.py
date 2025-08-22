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
import pandas as pd

def get_stock_history(symbol: str, period="6mo"):
    """Return historical OHLCV data"""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return hist.reset_index()

def get_stock_info(symbol: str):
    """Return latest price and fundamentals"""
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

