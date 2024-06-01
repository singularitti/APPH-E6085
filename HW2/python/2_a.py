#!/usr/bin/env python3
"""Problem 2(a).
Plot the density for this case versus the exact density.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh

alpha_1 = np.array([0.298073])
alpha_2 = np.array([0.298073, 1.242567])
alpha_3 = np.array([0.298073, 1.242567, 5.782948])
alpha_4 = np.array([0.298073, 1.242567, 5.782948, 38.474970])
my_path = os.path.abspath(__file__ + "/../../")


# Hamiltonian
def hamil_integrand(rr, a, b):
    return (
        4
        * np.pi
        * (-2 + 3 * a * rr - 2 * a**2 * rr**3)
        / rr
        * np.exp(-(a + b) * rr**2)
        * rr**2
    )


def hamiltonian(a, b):
    return quad(hamil_integrand, 0.0, np.inf, args=(a, b))[0]


# Overlap intergral
def overlap_integrand(rr, a, b):
    return 4 * np.pi * np.exp(-(a + b) * rr**2) * rr**2


def overlap(a, b):
    return quad(overlap_integrand, 0.0, np.inf, args=(a, b))[0]


def construct_eigval_problem(alpha):
    # Vectorize the table-like function
    vec_hamiltonian = np.vectorize(hamiltonian)
    h_ij = np.array([vec_hamiltonian(a, alpha) for a in alpha])

    # Vectorize the table-like function
    vec_overlap = np.vectorize(overlap)
    s_ij = np.array([vec_overlap(a, alpha) for a in alpha])

    return h_ij, s_ij


def eigenvalue_problem(alpha):
    h_ij, s_ij = construct_eigval_problem(alpha)

    # Generalized eigenvalue Problem H x = E S x
    eigvals, eigvecs = eigh(h_ij, s_ij, eigvals_only=False, type=1)
    return eigvecs[:, 0]  # Only return ground state eigenvector


def coefficient_matrix(alpha):
    eigvec = eigenvalue_problem(alpha)
    return np.outer(eigvec, eigvec)


def density(rr, alpha):
    vec_overlap_integrand = np.vectorize(
        lambda rr, a, b: np.exp(-(a + b) * rr**2)
    )  # This is not equal to overlap_integrand
    phi_phi_mat = np.array([vec_overlap_integrand(rr, a, alpha) for a in alpha])
    coeff_mat = coefficient_matrix(alpha)
    # Elementwise product, then sum all
    return np.multiply(coeff_mat, phi_phi_mat).sum()


def distribution_probability(rr, alpha):
    vec_overlap_integrand = np.vectorize(overlap_integrand)
    phi_phi_mat = np.array([vec_overlap_integrand(rr, a, alpha) for a in alpha])
    coeff_mat = coefficient_matrix(alpha)
    # Elementwise product, then sum all
    return np.multiply(coeff_mat, phi_phi_mat).sum()


# Exact solution
def exact_value(rr):
    return 2 * np.sqrt(2 / np.pi) * np.exp(-2 * rr)


def exact_density(rr):
    return exact_value(rr) ** 2


def exact_distribution_probability(rr):
    return 4 * np.pi * rr**2 * exact_density(rr)


# Plot
def plot():
    plt.style.use("classic")
    r = np.linspace(0, 3, endpoint=True, num=500)
    fig1 = plt.figure()
    plt.plot(r, [density(rr, alpha_1) for rr in r], label=r"$n=1$")
    plt.plot(r, [density(rr, alpha_2) for rr in r], label=r"$n=2$")
    plt.plot(r, [density(rr, alpha_3) for rr in r], label=r"$n=3$")
    plt.plot(r, [density(rr, alpha_4) for rr in r], label=r"$n=4$")
    plt.plot(r, [exact_density(rr) for rr in r], label=r"exact value")
    plt.xlabel(r"$r$ (Bohr)", fontsize=16)
    plt.ylabel(r"Density $|\psi(r)|^2$", fontsize=16)
    plt.legend(loc="best")

    fig2 = plt.figure()
    plt.plot(r, [distribution_probability(rr, alpha_1) for rr in r], label=r"$n=1$")
    plt.plot(r, [distribution_probability(rr, alpha_2) for rr in r], label=r"$n=2$")
    plt.plot(r, [distribution_probability(rr, alpha_3) for rr in r], label=r"$n=3$")
    plt.plot(r, [distribution_probability(rr, alpha_4) for rr in r], label=r"$n=4$")
    plt.plot(r, [exact_distribution_probability(rr) for rr in r], label=r"exact value")
    plt.xlabel(r"$r$ (Bohr)", fontsize=16)
    plt.ylabel(r"Radial probability distribution $|\chi(r)|^2$", fontsize=16)
    plt.legend(loc="best")
    # plt.show()
    fig1.savefig(my_path + "/images/pro_2_a_1.pdf")
    fig2.savefig(my_path + "/images/pro_2_a_2.pdf")


# plot()
