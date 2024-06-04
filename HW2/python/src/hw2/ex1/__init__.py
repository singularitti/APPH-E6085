#!/usr/bin/env python
"""Solve"""

import itertools

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, ishermitian, issymmetric

__all__ = [
    "basis",
    "hamiltonian",
    "overlap",
    "construct_problem",
    "solve_problem",
    "wavefunction",
]


def basis(𝛂):
    @np.vectorize
    def _at(r):
        return np.asarray([np.exp(-α * r**2) for α in 𝛂])

    return _at


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


def create_matrix(f, xs, ys):
    pairs = itertools.product(xs, ys)
    result = np.fromiter((f(x, y) for x, y in pairs), dtype=float)
    result = result.reshape(len(xs), len(ys))
    return result


def construct_problem(𝛂):
    Hᵢⱼ = create_matrix(hamiltonian, 𝛂, 𝛂)
    assert ishermitian(Hᵢⱼ, rtol=1e-8)
    # Vectorize the table-like function
    Sᵢⱼ = create_matrix(overlap, 𝛂, 𝛂)
    assert issymmetric(Sᵢⱼ, rtol=1e-8)
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


def wavefunction(𝛂, i=0):
    _, eigvecs = solve_problem(*construct_problem(𝛂))
    eigvec = eigvecs[i]
    𝐛 = basis(𝛂)

    @np.vectorize
    def _at(r):
        return np.dot(eigvec, 𝐛(r))

    return _at
