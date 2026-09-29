from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_market_data(
    tickers: List[str], 
    start_date: str = '2024-01-01', 
    end_date: str = '2025-01-01'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Mengunduh data harga penutupan dan menghitung return tahunan serta matriks kovariansi.
    """
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    
    # Pastikan urutan kolom konsisten dengan list tickers
    df = df[tickers]
    
    daily_returns = df.pct_change().dropna()
    mu = daily_returns.mean().values * 252       # Annualized Return
    sigma = daily_returns.cov().values * 252      # Annualized Covariance Matrix
    
    return mu, sigma, tickers