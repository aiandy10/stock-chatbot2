import yfinance as yf

symbol = "TCS.NS"  # TCS on NSE
stock = yf.Ticker(symbol)

# Get last closing price
hist = stock.history(period="1d")
print(f"{symbol} last close: {hist['Close'].iloc[-1]}")
