import pandas as pd
import requests
from typing import List
import logging

logger = logging.getLogger(__name__)

def fetch_nse_stocks() -> List[str]:
    """Fetch list of all NSE stocks"""
    try:
        # Fetch from NSE website
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        
        # Extract and clean symbols
        symbols = df['SYMBOL'].tolist()
        
        return sorted(symbols)  # Return sorted list for better UI
        
    except Exception as e:
        logger.error(f"Failed to fetch NSE stocks: {e}")
        # Fallback to cached list if fetch fails
        return load_cached_stocks()

def load_cached_stocks() -> List[str]:
    """Load cached NSE stock list as fallback"""
    try:
        cache_file = "data/nse_stocks.csv"
        df = pd.read_csv(cache_file)
        return df['SYMBOL'].tolist()
    except Exception:
        # Ultimate fallback - common stocks
        return ["TCS", "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK"]