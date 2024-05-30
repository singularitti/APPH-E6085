#!/usr/bin/env python3
"""Problem 1(b).
Plot the first excited state energy as a function of $n$.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from tabulate import tabulate

plt.style.use("classic")

alpha_1 = np.array([13])
alpha_2 = np.array([13, 1.96])
alpha_3 = np.array([13, 1.96, 0.44])
alpha_4 = np.array([13, 1.96, 0.44, 0.12])
my_path = os.path.abspath(__file__ + "/../../")


def eigenvalue_problem(alpha):
    # Hamiltonian
    def hamil_integrand(r, a, b):
        return (
            4
            * np.pi
            * (-1 + 3 * a * r - 2 * a**2 * r**3)
            / r
            * np.exp(-(a + b) * r**2)
            * r**2
        )

    def hamiltonian(a, b):
        return quad(hamil_integrand, 0.0, np.inf, args=(a, b))[0]

    # Vectorize the table-like function
    vec_hamiltonian = np.vectorize(hamiltonian)
    h_ij = [vec_hamiltonian(a, alpha) for a in alpha]  # Speed-up

    # print(tabulate(h_ij, tablefmt="latex"))

    # Overlap intergral
    def overlap_integrand(r, a, b):
        return 4 * np.pi * np.exp(-(a + b) * r**2) * r**2

    def overlap(a, b):
        return quad(overlap_integrand, 0.0, np.inf, args=(a, b))[0]

    # Vectorize the table-like function
    vec_overlap = np.vectorize(overlap)
    s_ij = [vec_overlap(a, alpha) for a in alpha]
    # print(tabulate(s_ij, tablefmt="latex"))

    # Generalized eigenvalue Problem H x = E S x
    eigvals, eigvecs = eigh(h_ij, s_ij, eigvals_only=False)
    return eigvals


# Plot
energy = [
    eigenvalue_problem(alpha_2)[1],
    eigenvalue_problem(alpha_3)[1],
    eigenvalue_problem(alpha_4)[1],
]
n = [2, 3, 4]

plt.figure()
plt.plot(n, energy)
plt.xticks(n)
plt.xlabel(r"$n$", fontsize=16)
plt.ylabel(r"$E$ (Hartree)", fontsize=16)
# plt.show()
plt.savefig(my_path + "/images/pro_1_b.pdf")
