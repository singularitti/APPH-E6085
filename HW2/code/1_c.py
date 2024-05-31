#!/usr/bin/env python3
"""Problem 1(c).
Plot the ground state eigenfunction as a function of $n$.
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
    h_ij = np.array([vec_hamiltonian(a, alpha) for a in alpha])

    # print(h_ij)
    # print(tabulate(h_ij, tablefmt="latex"))

    # Overlap intergral
    def overlap_integrand(r, a, b):
        return 4 * np.pi * np.exp(-(a + b) * r**2) * r**2

    def overlap(a, b):
        return quad(overlap_integrand, 0.0, np.inf, args=(a, b))[0]

    # Vectorize the table-like function
    vec_overlap = np.vectorize(overlap)
    s_ij = np.array([vec_overlap(a, alpha) for a in alpha])
    # print(s_ij)
    # print(tabulate(s_ij, tablefmt="latex"))

    # Generalized eigenvalue Problem H x = E S x
    eigvals, eigvecs = eigh(h_ij, s_ij, eigvals_only=False, type=1)
    return eigvecs[:, 0]  # Only return ground state eigenvector


def ground_state_wave_func(alpha):
    # Scipy doesn't give wanted sign of eigenvectors, so change sign manually
    def renormalize_eigvec(eigvec):
        if len([x for x in eigvec if x < 0]) > len(eigvec) / 2:
            return -eigvec
        else:
            return eigvec

    eigvec = eigenvalue_problem(alpha)
    eigvec = renormalize_eigvec(eigvec)
    basis = np.array([np.exp(-a * r**2) for a in alpha])
    return np.dot(eigvec, basis)


def exact_value(rr):
    return np.exp(-rr) / np.sqrt(np.pi)


# Plot
r = np.linspace(0, 3, endpoint=True, num=500)
plt.figure()
plt.plot(r, ground_state_wave_func(alpha_1), label=r"$n=1$")
plt.plot(r, -ground_state_wave_func(alpha_2), label=r"$n=2$")  # Must add minus sign
plt.plot(r, ground_state_wave_func(alpha_3), label=r"$n=3$")
plt.plot(r, ground_state_wave_func(alpha_4), label=r"$n=4$")
plt.plot(r, list(map(exact_value, r)), label=r"exact value")
plt.xlabel(r"$r$ (Bohr)", fontsize=16)
plt.ylabel(r"wave function", fontsize=16)
plt.ylim((-0.05, 1.05))
plt.legend(loc="best")
# plt.show()
plt.savefig(my_path + "/images/pro_1_c.pdf")
