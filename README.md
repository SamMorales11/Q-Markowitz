# Q-Markowitz: Hybrid Quantum-Classical Portfolio Optimization Platform

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Qiskit Core](https://img.shields.io/badge/Qiskit-1.x-61338F.svg)](https://qiskit.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Platform-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)

An institutional-grade quantitative finance platform mapping Markowitz Mean-Variance Portfolio Theory into Quadratic Unconstrained Binary Optimization (QUBO) and Ising Spin Glass Hamiltonians. The system executes the **Quantum Approximate Optimization Algorithm (QAOA)** via parameterized quantum circuits and benchmarks convergence against classical exact combinatorial eigensolvers on live equity market data.

---

## Visual Showcase

### Institutional Analytics Dashboard
Interactive execution terminal built with Streamlit and Plotly, styled with an institutional slate palette without generic UI components.

![Platform Overview](assets/dashboard_overview.png)
*Figure 1: Full-view optimization terminal displaying KPI telemetry, comparative metrics, capital allocation donut, and eigenstate probability spectrum.*

---

## Key Highlights

- **Formulation Rigor:** Direct algebraic mapping of Markowitz portfolio optimization into QUBO and Ising Hamiltonians using Pauli-Z spin operators ($x_i = \frac{I - Z_i}{2}$) without third-party black-box dependencies.
- **Hybrid VQA Engine:** Parameterized variational ansatz evaluation on statevector quantum backends coupled with gradient-free classical optimization (COBYLA).
- **Dual Runtime Interfaces:** Full experimental pipeline accessible via production-grade CLI (`main.py`) and a real-time web execution terminal (`app.py`).
- **NISQ Resilience Evaluation:** Empirical benchmarking of circuit depth scaling ($p=1$ vs. $p=2$), risk-aversion sensitivity ($q$), and depolarizing gate noise simulating physical hardware constraints.

---

## Mathematical Formulation

### 1. Markowitz Mean-Variance Portfolio Selection
The discrete portfolio selection problem optimizes the balance between expected asset return and portfolio variance under a strict asset count budget:

$$\min_{x \in \{0, 1\}^n} \quad q \cdot x^T \Sigma x - \mu^T x$$

Subject to the equality budget constraint:

$$\sum_{i=1}^n x_i = B$$

Where:
- $x_i \in \{0, 1\}$: Binary decision variable ($1$ if asset $i$ is included in the portfolio, $0$ otherwise).
- $\mu \in \mathbb{R}^n$: Annualized expected return vector derived from historical daily close prices ($252$ trading days).
- $\Sigma \in \mathbb{R}^{n \times n}$: Annualized sample covariance matrix representing asset co-movements.
- $q \in [0, 1]$: Risk-aversion coefficient ($q \to 0$: aggressive return-seeking, $q \to 1$: strict risk minimization).
- $B \in \mathbb{N}$: Target cardinality (exact number of assets to allocate).

---

### 2. QUBO & Ising Hamiltonian Mapping
To evaluate the constrained problem on quantum hardware, the equality budget constraint is integrated into the cost function via a quadratic penalty scalar $P$:

$$\mathcal{H}_{\text{QUBO}}(x) = q \sum_{i=1}^n \sum_{j=1}^n \Sigma_{ij} x_i x_j - \sum_{i=1}^n \mu_i x_i + P \left( \sum_{i=1}^n x_i - B \right)^2$$

Mapping binary variables $x_i \in \{0, 1\}$ to quantum spin operators $Z_i \in \{-1, +1\}$ using:

$$x_i = \frac{I - Z_i}{2}$$

Transforms the cost function into an **Ising Spin Glass Hamiltonian**:

$$H_C = \sum_{i=1}^n h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$$

The ground state (minimum eigenvalue eigenvector) of $H_C$ encodes the optimal asset combination.

---

## QAOA Algorithm Architecture

The Quantum Approximate Optimization Algorithm (QAOA) prepares a parameterized $n$-qubit quantum state through alternating layers of cost and mixer unitaries:

```text
|0⟩ ─── H ─── [ U(C, γ₁) ] ─── [ U(M, β₁) ] ─── ... ─── [ U(C, γₚ) ] ─── [ U(M, βₚ) ] ─── Measure
|0⟩ ─── H ─── [ U(C, γ₁) ] ─── [ U(M, β₁) ] ─── ... ─── [ U(C, γₚ) ] ─── [ U(M, βₚ) ] ─── Measure
|0⟩ ─── H ─── [ U(C, γ₁) ] ─── [ U(M, β₁) ] ─── ... ─── [ U(C, γₚ) ] ─── [ U(M, βₚ) ] ─── Measure
|0⟩ ─── H ─── [ U(C, γ₁) ] ─── [ U(M, β₁) ] ─── ... ─── [ U(C, γₚ) ] ─── [ U(M, βₚ) ] ─── Measure