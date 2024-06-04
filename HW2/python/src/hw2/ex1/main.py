#!/usr/bin/env python
"""Solve"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh


__all__ = [
    "basis",
    "hamiltonian",
    "overlap",
    "construct_problem",
    "solve_problem",
    "wavefunction",
]


def basis(𝛂, r):
    return np.asarray([np.exp(-α * r**2) for α in 𝛂])


def hamiltonian_integrand(r: float, αᵢ: float, αⱼ: float) -> float:
    """
    Compute the integrand for the Hamiltonian using the given parameters.

    This function calculates the integrand for the Hamiltonian with the formula:

    .. math::

        4\\pi r^2 \\left(\\frac{3\\alpha_j r - 2\\alpha_j^2 r^3 - 1}{r}\\right) e^{-(\\alpha_i + \\alpha_j)r^2}

    Args:
        r (float): The radial distance.
        αᵢ (float): Parameter :math:`\\alpha_i` in the integrand formula.
        αⱼ (float): Parameter :math:`\\alpha_j` in the integrand formula.

    Returns:
        float: The computed value of the integrand.
    """
    return 4 * np.pi * r * (3 * αⱼ * r - 2 * αⱼ**2 * r**3 - 1) * np.exp(-(αᵢ + αⱼ) * r**2)


@np.vectorize
def hamiltonian(αᵢ: float, αⱼ: float) -> float:
    return quad(hamiltonian_integrand, 0.0, np.inf, args=(αᵢ, αⱼ))[0]


def overlap_integrand(r: float, αᵢ: float, αⱼ: float) -> float:
    """
    Compute the overlap integrand using the given parameters.

    This function calculates the overlap integrand with the formula:

    .. math::

        4\\pi r^2 e^{-(\\alpha_i + \\alpha_j)r^2}

    Args:
        r (float): The radial distance.
        αᵢ (float): Parameter :math:`\\alpha_i` in the integrand formula.
        αⱼ (float): Parameter :math:`\\alpha_j` in the integrand formula.

    Returns:
        float: The computed value of the integrand.
    """
    return 4 * np.pi * r**2 * np.exp(-(αᵢ + αⱼ) * r**2)


@np.vectorize
def overlap(αᵢ: float, αⱼ: float) -> float:
    return quad(overlap_integrand, 0.0, np.inf, args=(αᵢ, αⱼ))[0]


def construct_problem(𝛂):
    Hᵢⱼ = np.asfarray([hamiltonian(α, 𝛂) for α in 𝛂])  # Speed-up
    # Vectorize the table-like function
    Sᵢⱼ = np.asfarray([overlap(α, 𝛂) for α in 𝛂])
    return Hᵢⱼ, Sᵢⱼ


def solve_problem(Hᵢⱼ, Sᵢⱼ):
    # Generalized eigenvalue Problem H x = E S x
    eigvals, eigvecs = eigh(Hᵢⱼ, Sᵢⱼ, eigvals_only=False)
    return eigvals, np.asarray(np.apply_along_axis(renormalize_eigvec, axis=0, arr=eigvecs))


# Scipy doesn't give wanted sign of eigenvectors, so change sign manually
def renormalize_eigvec(eigvec):
    if len([x for x in eigvec if x < 0]) > len(eigvec) / 2:
        return -eigvec
    else:
        return eigvec


def wavefunction(𝛂, r):
    _, eigvecs = solve_problem(*construct_problem(𝛂))
    𝐛 = basis(𝛂, r)
    return np.asarray([np.dot(eigvec, b) for eigvec, b in zip(eigvecs, 𝐛)])


if __name__ == "__main__":
    alpha_1 = np.asfarray([13])
    alpha_2 = np.asfarray([13, 1.96])
    alpha_3 = np.asfarray([13, 1.96, 0.44])
    alpha_4 = np.asfarray([13, 1.96, 0.44, 0.12])
    alphas = [alpha_1, alpha_2, alpha_3, alpha_4]
    nbasis = list(map(len, alphas))
    problems = [construct_problem(alpha) for alpha in alphas]
    energies = [solve_problem(*problem)[0][0] for problem in problems]

    plt.figure()
    plt.plot(nbasis, energies)
    plt.xticks(nbasis)
    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel(r"$E$ (Hartree)", fontsize=16)
    plt.show()
    print((construct_problem(alpha_4)[1][:, 0]))
    # my_path = os.path.abspath(__file__ + "/../../")
    # plt.savefig(my_path + "/images/pro_1_a.pdf")
