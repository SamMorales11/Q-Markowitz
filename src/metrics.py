from typing import Dict, Any, List
import numpy as np

def calculate_portfolio_metrics(
    binary_vector: np.ndarray, 
    mu: np.ndarray, 
    sigma: np.ndarray, 
    tickers: List[str], 
    risk_free_rate: float = 0.06
) -> Dict[str, Any]:
    """
    Menghitung bobot alokasi rata, return ekspektasi tahunan, volatilitas, dan Sharpe ratio.
    """
    sum_selected = np.sum(binary_vector)
    if sum_selected == 0:
        return {"return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0, "weights": {}}
        
    weights = binary_vector / sum_selected
    port_return = float(np.dot(weights, mu))
    port_volatility = float(np.sqrt(np.dot(weights.T, np.dot(sigma, weights))))
    
    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0.0
    allocation = {tickers[i]: round(float(weights[i]), 3) for i in range(len(tickers)) if weights[i] > 0}
    
    return {
        "return_annual": port_return,
        "volatility_annual": port_volatility,
        "sharpe_ratio": sharpe,
        "allocations": allocation
    }