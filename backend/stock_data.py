# backend/stock_data.py
import yfinance as yf

def get_stock_price(symbol: str) -> str:
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty:
            return f"No data available for {symbol}."
        last_close = hist['Close'].iloc[-1]
        return f"{symbol} last close: {last_close:.2f}"
    except Exception as e:
        return f"Error retrieving stock data for {symbol}: {e}"
