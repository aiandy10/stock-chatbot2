import yfinance as yf

def get_fundamentals(symbol: str):
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        "peRatio": info.get("trailingPE"),
        "pegRatio": info.get("pegRatio"),
        "debtToEquity": info.get("debtToEquity"),
        "currentRatio": info.get("currentRatio"),
        "eps": info.get("trailingEps"),
        "marketCap": info.get("marketCap"),
    }
def fetch_fundamentals(symbol: str):
    return get_stock_info(symbol)  # or whatever your main fundamentals function is
