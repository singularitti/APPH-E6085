#!/usr/bin/env python3
"""Problem 3(c).
Magnetic case.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.linalg import eigh
from scipy.special import erf

my_path = os.path.abspath(__file__ + "/../../")
plt.style.use('classic')

alpha = np.array([0.298073, 1.242567, 5.782948, 38.474970])
trial_1 = np.array([0.08088997, 0.2352165, 0.11090607, 0.06835098,
                    0.08088997, 0.2352165, 0.11090607, 0.06835098])  # eigenvector from 3(a)
trail_2 = np.array([0.46822748, 0.15162621, 0.07288406, 0.03556809,
                    -0.46822748, -0.15162621, -0.07288406, -0.03556809])  # eigenvector from 3(a)
iteration = 1
R = np.linspace(0.01, 3, num=100, endpoint=True)  # nuclear distance
r = np.linspace(-3, 3, num=100, endpoint=True)  # electron distance


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

    # fock term, 1 / | r_1 - r_2 |
    @staticmethod
    def electron_elctron_interaction(ai, aj, ak, al, Ra, Rb, Rc, Rd):
        ik = ai + ak
        jl = aj + al
        ijkl = ai + aj + ak + al
        Rp = (ai * Ra + ak * Rc) / ik
        Rq = (aj * Rb + al * Rd) / jl
        tmp_0 = (2 * np.pi ** (5 / 2) / ik / jl / np.sqrt(ijkl))
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

    def fock_matrix(self, R1, R2):
        # The order is crucial
        return \
            np.array(
                [self.electron_elctron_interaction(ai, aj, ak, al, Ra, Rb, Rc, Rd)
                 for Rd in [R1, R2]
                 for al in self.alpha
                 for Rb in [R1, R2]
                 for aj in self.alpha
                 for Rc in [R1, R2]
                 for ak in self.alpha
                 for Ra in [R1, R2]
                 for ai in self.alpha], dtype=np.float64).reshape(8, 8, 8, 8) - \
            np.array(
                [self.electron_elctron_interaction(ai, aj, ak, al, Ra, Rb, Rc, Rd)
                 for Rc in [R1, R2]
                 for ak in self.alpha
                 for Rb in [R1, R2]
                 for aj in self.alpha
                 for Rd in [R1, R2]
                 for al in self.alpha
                 for Ra in [R1, R2]
                 for ai in self.alpha], dtype=np.float64).reshape(8, 8, 8, 8)

    @jit
    def full_fock_matrix(self, R1, R2, coeff):
        """
        :param R1: position, scalar, float
        :param R2: position, scalar, float
        :param coeff: A numpy array, trial vec or eigvec
        :return: A 8x8 numpy matrix
        """
        return np.tensordot(np.outer(coeff, coeff), self.fock_matrix(R1, R2))

    def construct_eig_problem(self, R1, R2, coeff):
        """
        :param R1: position, scalar, float
        :param R2: position, scalar, float
        :param coeff: A numpy array, trial vec or eigvec
        :return: A tuple contains 2 8x8 numpy matrices
        """
        # The order of [Rn, aj, Rm, ai] is crucial, or matrix will not be positive definite
        t_ij = np.array([self.kinetic_energy(ai, aj, Rm, Rn)
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

        fock_ij = self.full_fock_matrix(R1, R2, coeff)

        s_ij = np.array([self.overlap(ai, aj, Rm, Rn)
                         for Rn in [R1, R2]
                         for aj in self.alpha
                         for Rm in [R1, R2]
                         for ai in self.alpha])

        n_n_ij = s_ij / abs(R1 - R2)

        return (t_ij + e_n_ij + n_n_ij).reshape(8, 8) + fock_ij, s_ij.reshape(8, 8)

    def solve_eigenvalue_problem(self, R1, R2, coeff):
        """
        :param R1: position, scalar, float
        :param R2: position, scalar, float
        :param coeff: A numpy array, trial vec or eigvec
        :return: A tuple contains a scalar and a numpy array
        """
        m_ij, s_ij = self.construct_eig_problem(R1, R2, coeff)

        # Generalized eigenvalue Problem (T+F+E+N) x = E S x
        eigvals, eigvecs = eigh(m_ij, s_ij, eigvals_only=False, type=1)
        # Only return ground state, both up and down spin are in the lowest
        # eigenvector
        return eigvals[0], eigvecs[:, 0]

    @staticmethod
    def ground_state_eig(mat_1, mat_2):
        eigvals, eigvecs = eigh(mat_1, mat_2, eigvals_only=False, type=1)
        return eigvals[0], eigvecs[:, 0]  # Only return ground state

    @staticmethod
    def excited_state_eig(mat_1, mat_2):
        eigvals, eigvecs = eigh(mat_1, mat_2, eigvals_only=False, type=1)
        return eigvals[1], eigvecs[:, 1]  # Only return first excited state

    # Compute energy
    def fock_energy(self, R1, R2, coeff):
        """
        :param R1:
        :param R2:
        :param coeff:
        :return: scalar
        """
        tmp = self.full_fock_matrix(R1, R2, coeff)  # 8x8 matrix
        return coeff.dot(tmp).dot(coeff)

    def total_energy(self, other, R1, R2, coeff):
        eigval1 = self.solve_eigenvalue_problem(R1, R2, coeff)[0]
        eigval2 = other.solve_eigenvalue_problem(R1, R2, coeff)[1]
        return eigval1 + eigval2 - self.fock_energy(R1, R2, coeff) - 1 / abs(R1 - R2)

    def self_consistency_loop(self, R1, R2, niter):
        eigvals_1 = np.zeros(niter)
        eigvals_2 = np.zeros(niter)
        eigvecs_1 = np.array([np.zeros(8) for i in range(niter)])
        eigvecs_2 = np.array([np.zeros(8) for i in range(niter)])
        total_es = np.zeros(niter)
        m_ij, s_ij = self.construct_eig_problem(R1, R2)  # Does not change
        fock_ij = self.full_fock_matrix(R1, R2, self.trial)  # Only Fock changes in each loop
        fock_e = self.trial.dot(fock_ij).dot(self.trial)

        eigvecs_2[0] = self.trial
        eigvals_2[0], new_eigvec = self.ground_state_eig(m_ij + fock_ij, s_ij)


h2 = HydrogenMolecule(alpha, trail_2)
eigval1, eigvec1 = h2.solve_eigenvalue_problem(0, 1, trail_2)
eigval2, eigvec2 = h2.solve_eigenvalue_problem(0, 1, eigvec1)
eigval1, eigvec1 = h2.solve_eigenvalue_problem(0, 1, eigvec2)
