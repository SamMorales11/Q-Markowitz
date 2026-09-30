import argparse
import numpy as np
from src.data_loader import fetch_market_data
from src.qubo_builder import build_portfolio_qp, qp_to_ising_operator
from src.solvers import solve_classical, solve_qaoa
from src.metrics import calculate_portfolio_metrics

def main():
    parser = argparse.ArgumentParser(description="Q-Markowitz: Quantum vs Classical Portfolio Optimization")
    parser.add_argument("--tickers", nargs="+", default=["BBCA.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"], help="List of stock tickers")
    parser.add_argument("--budget", type=int, default=2, help="Number of assets to select in the portfolio")
    parser.add_argument("--risk", type=float, default=0.5, help="Risk aversion factor q (0: aggressive, 1: conservative)")
    parser.add_argument("--reps", type=int, default=1, help="QAOA circuit layer depth (p)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Q-MARKOWITZ: QUANTUM PORTFOLIO OPTIMIZATION PIPELINE")
    print("=" * 60)
    print(f"Target Tickers : {args.tickers}")
    print(f"Budget         : {args.budget} assets")
    print(f"Risk Factor (q): {args.risk}")
    print(f"QAOA Layers (p): {args.reps}")
    print("-" * 60)
    
    # 1. Fetch Market Data
    print("[1/4] Mengunduh data historis & menghitung return / kovariansi...")
    mu, sigma, tickers = fetch_market_data(args.tickers)
    
    # 2. Build QP and Ising Operator
    print("[2/4] Menyusun formulasi QUBO & Ising Hamiltonian...")
    qp = build_portfolio_qp(tickers, mu, sigma, risk_factor=args.risk, budget=args.budget)
    operator, offset, _ = qp_to_ising_operator(qp)
    
    # 3. Solve with Classical Exact Solver
    print("[3/4] Menjalankan Classical Exact Solver (Baseline)...")
    res_classical = solve_classical(qp, tickers)
    
    # 4. Solve with QAOA
    print("[4/4] Menjalankan Quantum Solver (QAOA)...")
    res_qaoa = solve_qaoa(operator, qp, tickers, reps=args.reps)
    
    # Calculate Financial Metrics
    metrics_classical = calculate_portfolio_metrics(res_classical["binary_vector"], mu, sigma, tickers)
    metrics_qaoa = calculate_portfolio_metrics(res_qaoa["binary_vector"], mu, sigma, tickers)
    
    # Display Benchmark Table
    is_match = np.array_equal(res_classical["binary_vector"], res_qaoa["binary_vector"])
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 60)
    print(f"{'Metrik':<25} | {'Classical Exact':<15} | {'Quantum (QAOA)':<15}")
    print("-" * 60)
    print(f"{'Aset Terpilih':<25} | {str(res_classical['selected_tickers']):<15} | {str(res_qaoa['selected_tickers']):<15}")
    print(f"{'Binary Vector':<25} | {str(res_classical['binary_vector']):<15} | {str(res_qaoa['binary_vector']):<15}")
    print(f"{'Objective Value':<25} | {res_classical['objective_value']:<15.4f} | {res_qaoa['objective_value']:<15.4f}")
    print(f"{'Exp. Annual Return':<25} | {metrics_classical['return_annual']*100:<14.2f}% | {metrics_qaoa['return_annual']*100:<14.2f}%")
    print(f"{'Annual Volatility':<25} | {metrics_classical['volatility_annual']*100:<14.2f}% | {metrics_qaoa['volatility_annual']*100:<14.2f}%")
    print(f"{'Sharpe Ratio':<25} | {metrics_classical['sharpe_ratio']:<15.3f} | {metrics_qaoa['sharpe_ratio']:<15.3f}")
    print(f"{'Constraint Feasible':<25} | {'Ya':<15} | {'Ya' if res_qaoa['is_feasible'] else 'Tidak':<15}")
    print("-" * 60)
    print(f"Solusi Identik (Ground Truth Match): {'100% OPTIMAL' if is_match else 'SUB-OPTIMAL'}")
    print("=" * 60)

if __name__ == "__main__":
    main()