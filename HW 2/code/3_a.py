#!/usr/bin/env python3
"""Problem 3(a).
Compute the total energy as a function of distance between the protons.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.special import erf

my_path = os.path.abspath(__file__ + "/../../")
plt.style.use('classic')

alpha_he = np.array([0.298073, 1.242567, 5.782948, 38.474970])
trial = np.array([0.08088997, 0.2352165, 0.11090607, 0.06835098, 0.08088997, 0.2352165,
                  0.11090607, 0.06835098])
R = np.linspace(0.01, 5, num=500, endpoint=True)
r = np.linspace(-4, 4, num=500, endpoint=True)  # electron distance


class HydrogenMolecule(object):
    def __init__(self, a, t):
        self.alpha = a
        self.trial = t

    # - 1/2 \nabla^2
    @staticmethod
    def kinetic_energy(ai, aj, Rm, Rn):
        """
        First term on different atom sites, Rm =/ Rn
        :return: 8x8 matrix
        """
        tmp_1 = ai * aj / (ai + aj)
        tmp_2 = tmp_1 * (Rm - Rn) ** 2
        return (np.pi / (ai + aj)) ** (3 / 2) * \
               np.exp(-tmp_2) * tmp_1 * (3 - 2 * tmp_2)

    @staticmethod
    def overlap(ai, aj, Rm, Rn):
        """
        :param Rm: Rm = R1, R2
        :param Rn: Rn = R1, R2
        :return: 8x8 matrix
        """
        tmp_1 = ai * aj / (ai + aj)
        tmp_2 = tmp_1 * (Rm - Rn) ** 2
        return (np.pi / (ai + aj)) ** (3 / 2) * np.exp(-tmp_2)

    # 1/|r - R_l|
    @staticmethod
    def electron_nuclear_interaction(ai, aj, Rm, Rn, Rl):
        """
        Second & third term on different atom sites, Rm =/ Rn
        :param Rl: Rl=R1, R2
        :return: 8x8 matrix, called 2 times
        """
        Rp = (ai * Rm + aj * Rn) / (ai + aj)
        tmp_1 = ai * aj / (ai + aj) * (Rm - Rn) ** 2
        tmp_2 = np.sqrt((ai + aj) * (Rp - Rl) ** 2)
        # Must have this check, or will return NaN!
        if abs(tmp_2) < 1e-18:
            tmp_3 = 1
        else:
            tmp_3 = np.sqrt(np.pi) / 2 * 1 / tmp_2 * erf(tmp_2)
        return -(2 * np.pi / (ai + aj)) * np.exp(-tmp_1) * tmp_3

    # 1 / | R_1 - R_2 |
    def nuclear_nuclear_interaction(self, ai, aj, Rm, Rn, R1, R2):
        """
        Fifth term in Hamiltonian
        :param Rm: Rm = R1, R2
        :param Rn: Rn = R1, R2
        :return: 8x8 matrix
        """
        return self.overlap(ai, aj, Rm, Rn) / abs(R1 - R2)

    def construct_eig_problem(self, R1, R2):
        # The order of [Rn, aj, Rm, ai] is crucial, or it will not be positive definite
        kin_ij = np.array([self.kinetic_energy(ai, aj, Rm, Rn)
                           for Rn in [R1, R2]
                           for aj in self.alpha
                           for Rm in [R1, R2]
                           for ai in self.alpha])

        e_n_ij = np.array([self.electron_nuclear_interaction(ai, aj, Rm, Rn, R1)
                           for Rn in [R1, R2]
                           for aj in self.alpha
                           for Rm in [R1, R2]
                           for ai in self.alpha]) + \
                 np.array([self.electron_nuclear_interaction(ai, aj, Rm, Rn, R2)
                           for Rn in [R1, R2]
                           for aj in self.alpha
                           for Rm in [R1, R2]
                           for ai in self.alpha])

        n_n_ij = np.array([self.nuclear_nuclear_interaction(ai, aj, Rm, Rn, R1, R2)
                           for Rn in [R1, R2]
                           for aj in self.alpha
                           for Rm in [R1, R2]
                           for ai in self.alpha])

        s_ij = np.array([self.overlap(ai, aj, Rm, Rn)
                         for Rn in [R1, R2]
                         for aj in self.alpha
                         for Rm in [R1, R2]
                         for ai in self.alpha])

        return (kin_ij + e_n_ij + n_n_ij).reshape(8, 8), s_ij.reshape(8, 8)

    def solve_eigenvalue_problem(self, R1, R2):
        m_ij, s_ij = self.construct_eig_problem(R1, R2)

        # Generalized eigenvalue Problem (T+K+E+N) x = E S x
        eigvals, eigvecs = eigh(m_ij, s_ij, eigvals_only=False, type=1)
        return eigvals[0], eigvecs[:, 0]  # Only return ground state

    def total_energy(self, R1, R2):
        eigval = self.solve_eigenvalue_problem(R1, R2)[0]
        return 2 * eigval - 1 / abs(R1 - R2)

    # Compute Density
    def density(self, r, R1, R2, coeff):
        # This is not equal to overlap, because no integration is done
        phi_phi_mat = np.array(
            [np.exp(-ai * (r - Rm) ** 2 - aj * (r - Rn) ** 2)
             for Rn in [R1, R2]
             for aj in self.alpha
             for Rm in [R1, R2]
             for ai in self.alpha]).reshape(8, 8)
        coeff_mat = np.outer(coeff, coeff)  # 8x8 matrix
        return np.tensordot(phi_phi_mat, coeff_mat, axes=2)

    def test_if_orthnormal(self, R1, R2):
        eigvals, eigvecs = self.solve_eigenvalue_problem(R1, R2)
        coeff_matrix = np.outer(eigvecs, eigvecs)
        s_ij = self.construct_eig_problem(R1, R2)[1]
        return np.tensordot(coeff_matrix, s_ij)

# With He basis
# h2 = HydrogenMolecule(alpha_he, trial)
# e_total = [h2.total_energy(0, R2) for R2 in R]

# print distance corresponding to minimum eigenvalue
# print(R[np.argmin(e_total)])
# print(h2.solve_eigenvalue_problem(0, R[np.argmin(e_total)])[1])  # print eigvec for 3(b) problem
# print(h2.solve_eigenvalue_problem(0, R[np.argmin(e_total)])[1])  # print eigvec for 3(c) problem
# print('{:.12f}'.format(R[np.argmin(e_total)]))  # print min energy distance
# print(e_total[np.argmin(e_total)])  # print min energy


# def plot_e_total():
#     plt.figure()
#     plt.plot(R, e_total)
#     plt.ylim((-3, 8))
#     plt.xlabel("$| R_1 - R_2 |$ (Bohr)", fontsize=16)
#     plt.ylabel("$\\varepsilon$ (Hartree)", fontsize=16)
#     # plt.show()
#     plt.savefig(my_path + "/images/pro_3_a.pdf")

# With H basis
# h2_h = HydrogenMolecule(alpha_h, trial)
#
# plt.figure()
# plt.plot(R, [h2_h.total_energy(0, R2) for R2 in R])
# plt.ylim((-3, 8))
# plt.xlabel("$| R_1 - R_2 |$ (Bohr)", fontsize=16)
# plt.ylabel("$\\varepsilon$ (Hartree)", fontsize=16)
# plt.show()
