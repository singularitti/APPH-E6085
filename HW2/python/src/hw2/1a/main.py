#!/usr/bin/env python
"""Problem 1(a).
Plot the ground state energy as a function of $n$.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh


def hamiltonian_integrand(r: float, a: float, b: float) -> float:
    """
    Compute the integrand for the Hamiltonian using the given parameters.

    This function calculates the integrand for the Hamiltonian with the formula:

    .. math::

        4\\pi \\left( \\frac{-1 + 3a r - 2a^2 r^3}{r} \\right) e^{-(a + b)r^2} r^2

    Args:
        r (float): The radial distance.
        a (float): Parameter 'a' in the integrand formula.
        b (float): Parameter 'b' in the integrand formula.

    Returns:
        float: The computed value of the integrand.
    """
    return (
        4
        * np.pi
        * (-1 + 3 * a * r - 2 * a**2 * r**3)
        / r
        * np.exp(-(a + b) * r**2)
        * r**2
    )


@np.vectorize
def hamiltonian(a, b):
    return quad(hamiltonian_integrand, 0.0, np.inf, args=(a, b))[0]


def overlap_integrand(r: float, a: float, b: float) -> float:
    """
    Compute the overlap integrand using the given parameters.

    This function calculates the overlap integrand with the formula:

    .. math::

        4 \\pi e^{-(a + b)r^2} r^2

    Args:
        r (float): The radial distance.
        a (float): Parameter 'a' in the integrand formula.
        b (float): Parameter 'b' in the integrand formula.

    Returns:
        float: The computed value of the integrand.
    """
    return 4 * np.pi * np.exp(-(a + b) * r**2) * r**2


@np.vectorize
def overlap(a, b):
    return quad(overlap_integrand, 0.0, np.inf, args=(a, b))[0]


def eigenvalue_problem(alpha):
    h_ij = np.asfarray([hamiltonian(a, alpha) for a in alpha])  # Speed-up
    # Vectorize the table-like function
    s_ij = np.asfarray([overlap(a, alpha) for a in alpha])
    # Generalized eigenvalue Problem H x = E S x
    eigvals, eigvecs = eigh(h_ij, s_ij, eigvals_only=False)
    return eigvals, eigvecs


if __name__ == "__main__":
    alpha_1 = np.asfarray([13])
    alpha_2 = np.asfarray([13, 1.96])
    alpha_3 = np.asfarray([13, 1.96, 0.44])
    alpha_4 = np.asfarray([13, 1.96, 0.44, 0.12])
    nbasis = list(map(len, [alpha_1, alpha_2, alpha_3, alpha_4]))
    energies = [
        eigenvalue_problem(alpha_1)[0][0],
        eigenvalue_problem(alpha_2)[0][0],
        eigenvalue_problem(alpha_3)[0][0],
        eigenvalue_problem(alpha_4)[0][0],
    ]

    plt.figure()
    plt.plot(nbasis, energies)
    plt.xticks(nbasis)
    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel(r"$E$ (Hartree)", fontsize=16)
    plt.show()
    print((eigenvalue_problem(alpha_4)[1][:, 0]))
    # my_path = os.path.abspath(__file__ + "/../../")
    # plt.savefig(my_path + "/images/pro_1_a.pdf")
