# Q-Markowitz: Benchmarking QAOA vs. Classical Solvers for Portfolio Optimization

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-61338F.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end quantum finance project formulating Markowitz Mean-Variance Portfolio Optimization as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solving it using the hybrid quantum-classical **Quantum Approximate Optimization Algorithm (QAOA)**, benchmarked against classical exact eigensolvers.

---

## Table of Contents
- [Overview](#overview)
- [Financial Formulation](#financial-formulation)
- [QUBO & Ising Hamiltonian Mapping](#qubo--ising-hamiltonian-mapping)
- [QAOA Architecture](#qaoa-architecture)
- [Visualizations & Circuit](#visualizations--circuit)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Key Takeaways](#key-takeaways)

---

## Overview

Modern Portfolio Theory (MPT), introduced by Harry Markowitz, seeks to select an optimal asset allocation that maximizes expected return for a targeted level of risk. When selecting an exact subset of $B$ assets out of $N$ candidates, the problem manifests as a NP-hard combinatorial quadratic optimization task.

This project demonstrates:
1. Retrieval and processing of real market stock data (IDX equities: `BBCA`, `TLKM`, `ASII`, `BMRI`) via `yfinance`.
2. Formulation of the quadratic risk-return objective and linear budget constraints into a canonical QUBO matrix.
3. Conversion of QUBO into an Ising Hamiltonian operator.
4. Execution of a parameterized QAOA variational ansatz ($p=1$) using a classical optimizer (COBYLA) and statevector probability sampling.
5. Verification of quantum solutions against a classical Exact Eigensolver baseline.

---

## Financial Formulation

The discrete Markowitz portfolio selection problem is expressed as:

$$\min_{x \in \{0, 1\}^n} \quad q \cdot x^T \Sigma x - \mu^T x$$

Subject to the budget constraint:

$$\sum_{i=1}^n x_i = B$$

Where:
- $x_i \in \{0, 1\}$: Binary decision variable ($1$ if asset $i$ is selected, $0$ otherwise).
- $\mu \in \mathbb{R}^n$: Vector of expected annualized asset returns.
- $\Sigma \in \mathbb{R}^{n \times n}$: Annualized covariance matrix capturing asset volatility and co-movement.
- $q \in [0, 1]$: Risk-aversion parameter ($q \to 0$: return-seeking, $q \to 1$: risk-averse).
- $B \in \mathbb{N}$: Target budget (exact count of assets to include).

---

## QUBO & Ising Hamiltonian Mapping

To evaluate this constrained problem on quantum hardware, the equality constraint is incorporated into the cost function via a quadratic penalty parameter $P$:

$$\text{Objective}(x) = q \cdot x^T \Sigma x - \mu^T x + P \left( \sum_{i=1}^n x_i - B \right)^2$$

The binary variables $x_i \in \{0, 1\}$ are subsequently mapped to spin variables $s_i \in \{-1, +1\}$ using the Pauli-Z transformation:

$$x_i = \frac{I - Z_i}{2}$$

Substituting this identity converts the objective function into an **Ising Spin Glass Hamiltonian**:

$$H_C = \sum_{i} h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset}$$

The ground state (minimum eigenvalue eigenvector) of $H_C$ encodes the optimal asset combination.

---

## QAOA Architecture

The Quantum Approximate Optimization Algorithm operates via a parameterized quantum circuit of depth $p$:

1. **Initial State:** Uniform superposition across all $2^n$ basis states:
   $$\vert{}\psi_0\rangle = \vert{}+\rangle^{\otimes n} = H^{\otimes n} \vert{}0\rangle^{\otimes n}$$
2. **Cost Unitary:** Evolves the state under the cost Hamiltonian $H_C$:
   $$U(C, \gamma) = e^{-i \gamma H_C}$$
3. **Mixer Unitary:** Drives transitions between computational states using transverse field operators:
   $$U(B, \beta) = e^{-i \beta H_M}, \quad H_M = \sum_{i=1}^n X_i$$
4. **Variational State ($p$-layers):**
   $$\vert{}\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{k=1}^p U(B, \beta_k) U(C, \gamma_k) \vert{}\psi_0\rangle$$
5. **Classical Optimization Loop:** A classical optimizer (COBYLA, 100 iterations) iteratively updates $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ to minimize the energy expectation value $E(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \langle \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \vert{} H_C \vert{} \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle$.

---

## Visualizations & Circuit

### 1. QAOA Parameterized Circuit ($p=1$)
Parameterized quantum circuit diagram displaying initial Hadamard transformations, entangling phase-separation gates ($ZZ$), and single-qubit mixer rotations ($R_x$):

![QAOA Quantum Circuit](assets/qaoa_circuit.png)

### 2. Quantum State Probability Distribution
Sampling measurement probabilities over all candidate basis states. The target ground-state bitstring emerges with dominant measurement probability:

![Probability Distribution](assets/probability_distribution.png)

---

## Benchmark Results

| Metric | Classical Solver (Exact Eigensolver) | Quantum Solver (QAOA, $p=1$) |
| :--- | :--- | :--- |
| **Algorithm Type** | Exact Matrix Diagonalization | Variational Quantum Algorithm (VQE/QAOA) |
| **Selected Portfolio** | `['TLKM.JK', 'BMRI.JK']` | `['TLKM.JK', 'BMRI.JK']` |
| **Binary Vector ($x$)** | `[0, 1, 0, 1]` | `[0, 1, 0, 1]` |
| **Objective Value ($f_{\text{val}}$)** | `-0.0412` | `-0.0412` |
| **Constraint Status** | Valid ($\sum x_i = 2$) | Valid ($\sum x_i = 2$) |
| **Optimality Gap** | `0.00%` (Ground Truth) | `0.00%` (Optimal Match) |
| **Expected Annual Return** | `14.82%` | `14.82%` |
| **Annualized Volatility** | `18.45%` | `18.45%` |
| **Sharpe Ratio ($R_f=6\%$)** | `0.478` | `0.478` |

> *Note: Metrics are calculated from real historical market data over the 2024 trading window. Results demonstrate zero optimality gap for $N=4, B=2$ at depth $p=1$.*

---

## Project Structure

```text
Q-Markowitz/
│
├── assets/                          # Generated plots and circuit diagrams
│   ├── qaoa_circuit.png             # Decomposed QAOA circuit plot
│   └── probability_distribution.png # Measurement probability bar chart
│
├── notebooks/
│   └── q_markowitz_qaoa.ipynb       # Main end-to-end experimental notebook
│
├── .gitignore                       # Ignored venv, cache, and temp files
├── LICENSE                          # MIT License
├── README.md                        # Documentation and benchmarking report
└── requirements.txt                 # Pinned dependencies