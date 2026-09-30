from typing import Dict, Any, List
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import SparsePauliOp

def solve_classical(qp: QuadraticProgram, tickers: List[str]) -> Dict[str, Any]:
    """
    Menemukan solusi optimal menggunakan Classical Exact Eigensolver sebagai ground truth.
    """
    exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = exact_solver.solve(qp)
    
    binary_solution = np.array([int(val) for val in result.x])
    selected_tickers = [tickers[i] for i, val in enumerate(binary_solution) if val == 1]
    
    return {
        "solver": "Classical Exact Eigensolver",
        "binary_vector": binary_solution,
        "selected_tickers": selected_tickers,
        "objective_value": float(result.fval),
        "status": str(result.status)
    }

def solve_qaoa(
    operator: SparsePauliOp, 
    qp: QuadraticProgram, 
    tickers: List[str], 
    reps: int = 1, 
    maxiter: int = 100
) -> Dict[str, Any]:
    """
    Mengeksekusi QAOA secara hybrid quantum-classical menggunakan statevector simulation.
    """
    ansatz = QAOAAnsatz(cost_operator=operator, reps=reps)
    
    def cost_func(params):
        circuit = ansatz.assign_parameters(params)
        return Statevector.from_instruction(circuit).expectation_value(operator).real
    
    initial_params = np.ones(ansatz.num_parameters) * 0.5
    opt_result = minimize(cost_func, initial_params, method='COBYLA', options={'maxiter': maxiter})
    
    # Evaluasi statevector optimal
    optimal_circuit = ansatz.assign_parameters(opt_result.x)
    final_state = Statevector.from_instruction(optimal_circuit)
    probabilities = final_state.probabilities_dict()
    
    best_bitstring = max(probabilities, key=probabilities.get)
    # Qiskit menggunakan konvensi little-endian (bitstring dibalik)
    binary_solution = np.array([int(b) for b in reversed(best_bitstring)])
    obj_val = float(qp.objective.evaluate(binary_solution))
    selected_tickers = [tickers[i] for i, val in enumerate(binary_solution) if val == 1]
    
    return {
        "solver": f"QAOA (p={reps})",
        "binary_vector": binary_solution,
        "selected_tickers": selected_tickers,
        "objective_value": obj_val,
        "optimal_parameters": opt_result.x,
        "probabilities": probabilities,
        "is_feasible": qp.is_feasible(binary_solution)
    }