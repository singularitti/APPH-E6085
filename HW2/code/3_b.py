#!/usr/bin/env python3
"""Problem 3(b).
Plot the energy as a function of distance
"""

import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.linalg import eigh
from scipy.special import erf

mod = importlib.import_module("3_a")

my_path = os.path.abspath(__file__ + "/../../")
plt.style.use("classic")

alpha = np.array([0.298073, 1.242567, 5.782948, 38.474970])
trial = np.array(
    [
        0.08088997,
        0.2352165,
        0.11090607,
        0.06835098,
        0.08088997,
        0.2352165,
        0.11090607,
        0.06835098,
    ]
)  # eigenvector from 3(a)
iteration = 45
R = np.linspace(0.01, 3, num=250, endpoint=True)  # nuclear distance
r = np.linspace(-3, 3, num=500, endpoint=True)  # electron distance


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
        return (np.pi / (ai + aj)) ** (3 / 2) * np.exp(-tmp_2) * tmp_1 * (3 - 2 * tmp_2)

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
        if tmp_2 < 1e-18:
            tmp_3 = 1
        else:
            tmp_3 = np.sqrt(np.pi) / 2 * 1 / tmp_2 * erf(tmp_2)
        return -(2 * np.pi / (ai + aj)) * np.exp(-tmp_1) * tmp_3

    # 1 / | R_1 - R_2 |
    def nuclear_nuclear_interaction(self, ai, aj, Rm, Rn, R1, R2):
        """
        No need to call, just use overlap is okay, defined just for testing
        Fifth term in Hamiltonian
        :param Rm: Rm = R1, R2
        :param Rn: Rn = R1, R2
        :return: 8x8 matrix
        """
        return self.overlap(ai, aj, Rm, Rn) / abs(R1 - R2)

    # Hartree term, 1 / | r_1 - r_2 |
    @staticmethod
    def electron_elctron_interaction(ai, aj, ak, al, Ra, Rb, Rc, Rd):
        ik = ai + ak
        jl = aj + al
        ijkl = ai + aj + ak + al
        Rp = (ai * Ra + ak * Rc) / ik
        Rq = (aj * Rb + al * Rd) / jl
        tmp_0 = 2 * np.pi ** (5 / 2) / ik / jl / np.sqrt(ijkl)
        tmp_1 = ai * ak / ik
        tmp_2 = aj * al / jl
        tmp_3 = np.exp(-tmp_1 * (Ra - Rc) ** 2 - tmp_2 * (Rb - Rd) ** 2)
        tmp_4 = np.sqrt(ik * jl / ijkl * (Rp - Rq) ** 2)
        # Must have this check, or will return NaN!
        if tmp_4 < 1e-18:
            tmp_5 = 1
        else:
            tmp_5 = np.sqrt(np.pi) / 2 * 1 / tmp_4 * erf(tmp_4)
        return tmp_0 * tmp_3 * tmp_5

    def hartree_matrix(self, R1, R2):
        # The order is crucial
        return np.array(
            [
                self.electron_elctron_interaction(ai, aj, ak, al, Ra, Rb, Rc, Rd)
                for Rd in [R1, R2]
                for al in self.alpha
                for Rb in [R1, R2]
                for aj in self.alpha
                for Rc in [R1, R2]
                for ak in self.alpha
                for Ra in [R1, R2]
                for ai in self.alpha
            ],
            dtype=np.float64,
        ).reshape(8, 8, 8, 8)

    @jit
    def full_hartree_matrix(self, R1, R2, coeff):
        """
        :param R1: position, scalar, float
        :param R2: position, scalar, float
        :param coeff: A numpy array, trial vec or eigvec
        :return: A 8x8 numpy matrix
        """
        return np.tensordot(np.outer(coeff, coeff), self.hartree_matrix(R1, R2))

    def construct_eig_problem(self, R1, R2):
        """
        :param R1: position, scalar, float
        :param R2: position, scalar, float
        :return: A tuple contains 2 8x8 numpy matrices
        """
        # The order of [Rn, aj, Rm, ai] is crucial, or matrix will not be positive definite
        t_ij = np.array(
            [
                self.kinetic_energy(ai, aj, Rm, Rn)
                for Rn in [R1, R2]
                for aj in self.alpha
                for Rm in [R1, R2]
                for ai in self.alpha
            ]
        )

        e_n_ij = np.array(
            [
                self.electron_nuclear_interaction(ai, aj, Rm, Rn, R1)
                for Rn in [R1, R2]
                for aj in self.alpha
                for Rm in [R1, R2]
                for ai in self.alpha
            ]
        ) + np.array(
            [
                self.electron_nuclear_interaction(ai, aj, Rm, Rn, R2)
                for Rn in [R1, R2]
                for aj in self.alpha
                for Rm in [R1, R2]
                for ai in self.alpha
            ]
        )

        s_ij = np.array(
            [
                self.overlap(ai, aj, Rm, Rn)
                for Rn in [R1, R2]
                for aj in self.alpha
                for Rm in [R1, R2]
                for ai in self.alpha
            ]
        )

        n_n_ij = s_ij / abs(R1 - R2)

        return (t_ij + e_n_ij + n_n_ij).reshape(8, 8), s_ij.reshape(8, 8)

    @staticmethod
    def ground_state_eig(mat_1, mat_2):
        eigvals, eigvecs = eigh(mat_1, mat_2, eigvals_only=False, type=1)
        return eigvals[0], eigvecs[:, 0]  # Only return ground state

    def self_consistency_loop(self, R1, R2, niter):
        eigvals = np.zeros(niter)
        eigvecs = np.array([np.zeros(8) for i in range(niter)])
        total_es = np.zeros(niter)
        m_ij, s_ij = self.construct_eig_problem(R1, R2)  # Does not change
        hartree_ij = self.full_hartree_matrix(
            R1, R2, self.trial
        )  # Only Hartree changes in each loop
        hartree_e = self.trial.dot(hartree_ij).dot(self.trial)

        eigvecs[0] = self.trial
        eigvals[0], new_eigvec = self.ground_state_eig(m_ij + hartree_ij, s_ij)
        total_es[0] = 2 * eigvals[0] - hartree_e - 1 / abs(R1 - R2)

        # To make eigval and eigvec match, eigenvector and corresponding eigenvalue have same index!
        for i in range(niter):
            eigvecs[i] = new_eigvec

            eigvals[i], new_eigvec = self.ground_state_eig(
                m_ij + self.full_hartree_matrix(R1, R2, eigvecs[i]), s_ij
            )

            hartree_ij = self.full_hartree_matrix(R1, R2, eigvecs[i])
            hartree_e = eigvecs[i].dot(hartree_ij).dot(eigvecs[i])
            total_es[i] = 2 * eigvals[i] - hartree_e - 1 / abs(R1 - R2)

        # Only return convergent result
        return eigvals[-1], total_es[-1], eigvecs[-1]

    # Compute Density
    def density(self, r, R1, R2, coeff):
        # This is not equal to overlap, because no integration is done
        phi_phi_mat = np.array(
            [
                np.exp(-ai * (r - Rm) ** 2 - aj * (r - Rn) ** 2)
                for Rn in [R1, R2]
                for aj in self.alpha
                for Rm in [R1, R2]
                for ai in self.alpha
            ]
        ).reshape(8, 8)
        coeff_mat = np.outer(coeff, coeff)  # 8x8 matrix
        return np.tensordot(phi_phi_mat, coeff_mat, axes=2)


h2 = HydrogenMolecule(alpha, trial)

non_interact_h2 = mod.HydrogenMolecule(alpha, trial)

# last eigvals, last total_es, last eigvecs at each distance | R_1 - R_2 |
lst_eigvals = np.zeros(R.size)
lst_e = np.zeros(R.size)
lst_eigvecs = np.array([np.zeros(8) for i in range(R.size)])
for i, R2 in np.ndenumerate(R):
    lst_eigvals[i], lst_e[i], lst_eigvecs[i] = h2.self_consistency_loop(
        0, R2, iteration
    )

print(("{:.12f}".format(R[np.argmin(lst_e)])))  # print min energy distance
print((lst_e[np.argmin(lst_e)]))  # print min energy

interact_den = np.array(
    [h2.density(rr, 0, R[np.argmin(lst_e)], lst_eigvecs[np.argmin(lst_e)]) for rr in r]
)
non_interact_eigenvec = non_interact_h2.solve_eigenvalue_problem(0, 1.02)[1]
non_interact_den = np.array(
    [non_interact_h2.density(rr, 0, 1.02, non_interact_eigenvec) for rr in r]
)

fig1 = plt.figure()
plt.plot(R, lst_e)
plt.ylim((-2, 8))
plt.xlabel("$| R_1 - R_2 |$ (Bohr)", fontsize=16)
plt.ylabel("$\\varepsilon$ (Hartree)", fontsize=16)
fig1.savefig(my_path + "/images/pro_3_b_1.pdf")

fig2 = plt.figure()
plt.plot(r, interact_den, label="interacting")
plt.plot(r, non_interact_den, label="non-interacting")
plt.xlabel("$R_1 - R_2$ (Bohr)", fontsize=16)
plt.ylabel("Density $| \psi(r) |^2$", fontsize=16)
plt.legend(loc="best")
# plt.show()
fig2.savefig(my_path + "/images/pro_3_b_2.pdf")
