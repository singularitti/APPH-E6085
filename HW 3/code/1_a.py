#!/usr/bin/env python3
"""
Problem 1(a).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh

alpha = np.array([0.298073, 1.242567, 5.782948, 38.474970])
trial = np.array([0.09598226, 0.16365395, 0.18675064, 0.07186526])
niter = 10

my_path = os.path.abspath(__file__ + "/../../")
plt.style.use('classic')


class HydrogenAtom(object):
    def __init__(self, a, t, n):
        self.alpha = a
        self.trial = t
        self.niter = n
        self.eigvals = np.zeros(self.niter)
        self.eigvecs = np.array([np.zeros(4) for i in range(self.niter)])
        self.energy = np.zeros(self.niter)

    # - 1/2 \nabla^2 - 1/r
    @staticmethod
    def hamil_integrand(r, ai, aj):
        return 4 * np.pi * (-1 + 3 * ai * r - 2 * ai ** 2 * r ** 3) \
               / r * np.exp(-(ai + aj) * r ** 2) * r ** 2

    def hamiltonian(self, ai, aj):
        return quad(self.hamil_integrand, 0.0, np.inf, args=(ai, aj))[0]

    # Overlap intergral
    @staticmethod
    def overlap_integrand(rr, ai, aj):
        return 4 * np.pi * np.exp(-(ai + aj) * rr ** 2) * rr ** 2

    def overlap(self, ai, aj):
        return quad(self.overlap_integrand, 0.0, np.inf, args=(ai, aj))[0]

    # Hartree potential
    @staticmethod
    def hartree_integration(ai, am, aj, an):
        """
        :param ai: outer integration
        :param am: inner integration, potential term
        :param aj: outer integration
        :param an: inner integration, potential term
        :return: function
        """
        return 2 * np.pi ** (5 / 2) / (ai + aj) / (am + an) / np.sqrt(ai + am + aj + an)

    def hartree_matrix(self, alpha: np.ndarray) -> np.ndarray:
        har = np.array([self.hartree_integration(a_i, a_m, a_j, a_n)
                        for a_i in alpha for a_j in alpha
                        for a_m in alpha for a_n in alpha]).reshape(4, 4, 4, 4)
        # The order of [a_i, a_m, a_j, a_n] is crucial
        return har

    @staticmethod
    def coefficient_matrix(coeff):
        return np.outer(coeff, coeff)

    def hartree_matrix_tensor_contract(self, coeff: np.ndarray) -> np.ndarray:
        """
        Hartree term, matrix representation
        :param coeff: 1x4 array
        :return: 4x4 matrix
        """
        return np.tensordot(self.coefficient_matrix(coeff),
                            self.hartree_matrix(self.alpha), axes=2)

    def hartree_energy(self, eigvec: np.ndarray) -> np.float64:
        return eigvec.dot(self.hartree_matrix_tensor_contract(eigvec)).dot(eigvec)

    # Exchange potential
    @staticmethod
    def electron_density(r, alpha: np.ndarray, coeff: np.ndarray):
        """
        Exchange integrand
        :param r: variable
        :param alpha:
        :param coeff: eigenvector
        :return: scalar, the sum of 4x4 matrix elements
        """
        tmp_1 = np.array([[np.exp(-(a_i + a_j) * r ** 2)
                           for a_i in alpha] for a_j in alpha])
        tmp_2 = np.outer(coeff, coeff)
        return np.tensordot(tmp_1, tmp_2)

    def exchange_integration(self, alpha: np.ndarray, coeff: np.ndarray):
        """
        N_{ij}
        :param alpha:
        :param coeff: eigenvector
        :return: 4x4 matrix
        """
        tmp = lambda r, a_i, a_j: -4 * np.pi * (3 / np.pi) ** (1 / 3) * r ** 2 * \
                                  np.exp(-(a_i + a_j) * r ** 2) * \
                                  (self.electron_density(r, alpha, coeff)) ** (1 / 3)
        return np.array(
            [quad(tmp, 0, np.inf, args=(a_i, a_j))[0] for a_i in alpha for a_j in alpha]).reshape(4, 4)

    def exchange_energy(self, eigvec: np.ndarray) -> np.float64:
        coeff_mat = self.coefficient_matrix(eigvec)
        energy_mat = 3 / 4 * self.exchange_integration(self.alpha, eigvec)
        return np.tensordot(coeff_mat, energy_mat, axes=2)

    def exchange_potential(self, coeff: np.ndarray):
        # tmp = lambda r: -4 * np.pi * r ** 2 * (3 / np.pi) ** (1 / 3) * \
        #                 (self.electron_density(r, self.alpha, coeff)) ** (4 / 3)
        # return quad(tmp, 0, np.inf)[0]
        return 4 / 3 * self.exchange_energy(coeff)  # equivalent to above

    # Now build the problem
    def construct_eigval_problem(self, alpha: np.ndarray, coeff: np.ndarray) -> (np.ndarray, np.ndarray):
        # Vectorize the table-like function
        vec_hamiltonian = np.vectorize(self.hamiltonian)
        h_ij = np.array([vec_hamiltonian(a, alpha) for a in alpha])

        vec_overlap = np.vectorize(self.overlap)
        s_ij = np.array([vec_overlap(a, alpha) for a in alpha])

        k_ij = self.hartree_matrix_tensor_contract(coeff)

        n_ij = self.exchange_integration(alpha, coeff)

        return h_ij, k_ij, n_ij, s_ij

    def solve_eigenvalue_problem(self, alpha, coeff):
        h_ij, k_ij, n_ij, s_ij = self.construct_eigval_problem(alpha, coeff)

        # Generalized eigenvalue Problem (H+K+N) x = E S x
        eigvals, eigvecs = eigh(h_ij + k_ij + n_ij, s_ij,
                                eigvals_only=False, type=1)
        return eigvals[0], eigvecs[:, 0]  # Only return ground state

    # Final Energy
    def total_energy(self, eigval, eigvec: np.ndarray):
        return eigval - self.hartree_energy(eigvec) + \
               self.exchange_energy(eigvec) - self.exchange_potential(eigvec)

    def loop_and_calculate(self, niter):
        # Be sure to make eigval and eigvec match!
        self.eigvecs[0] = self.trial
        self.eigvals[0] = self.solve_eigenvalue_problem(alpha, trial)[0]

        for i in range(1, niter):
            self.eigvecs[i] = self.solve_eigenvalue_problem(
                alpha, self.eigvecs[i - 1])[1]
            self.eigvals[i] = self.solve_eigenvalue_problem(
                alpha, self.eigvecs[i - 1])[0]

        return self.eigvals, self.eigvecs


h = HydrogenAtom(alpha, trial, niter)
eigvals, eigvecs = h.loop_and_calculate(niter)


# print(eigvals[-1])

def my_plot_1():
    iter_list = range(niter)
    plt.figure()
    plt.plot(iter_list, [h.total_energy(eigvals[i], eigvecs[i])
                         for i in range(niter)])
    plt.show()


# my_plot_1()
print(eigvals[-1])
print(h.total_energy(eigvals[-1], eigvecs[-1]))
print(-eigvals[-1] - h.hartree_energy(eigvecs[-1]) + h.exchange_energy(eigvecs[-1]) - h.exchange_potential(eigvecs[-1]))
