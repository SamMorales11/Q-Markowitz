from typing import List, Tuple
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp

def build_portfolio_qp(
    tickers: List[str], 
    mu: list, 
    sigma: list, 
    risk_factor: float = 0.5, 
    budget: int = 2
) -> QuadraticProgram:
    """
    Membangun model Quadratic Program Markowitz dengan binary decision variables.
    """
    num_assets = len(tickers)
    qp = QuadraticProgram(name="Q_Markowitz_Optimization")
    
    for ticker in tickers:
        qp.binary_var(name=ticker)
        
    linear_terms = {tickers[i]: -float(mu[i]) for i in range(num_assets)}
    quadratic_terms = {}
    
    for i in range(num_assets):
        for j in range(num_assets):
            val = risk_factor * float(sigma[i][j])
            if val != 0:
                quadratic_terms[(tickers[i], tickers[j])] = val
                
    qp.minimize(linear=linear_terms, quadratic=quadratic_terms)
    
    qp.linear_constraint(
        linear={ticker: 1 for ticker in tickers},
        sense="==",
        rhs=budget,
        name="budget_constraint"
    )
    return qp

def qp_to_ising_operator(qp: QuadraticProgram) -> Tuple[SparsePauliOp, float, QuadraticProgram]:
    """
    Mengonversi QuadraticProgram menjadi Ising Hamiltonian operator.
    """
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    operator, offset = qubo.to_ising()
    return operator, offset, qubo