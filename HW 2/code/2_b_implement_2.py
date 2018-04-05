#!/usr/bin/env python3
"""Problem 2(b).
Self-consistent iteration.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
from matplotlib.ticker import MultipleLocator
from scipy.integrate import quad
from scipy.linalg import eigh

mod = __import__("2_a")

alpha = np.array([0.298073, 1.242567, 5.782948, 38.474970])
trial = np.array([0.08704173, 0.52509737, 0.51920929, 0.31233737])
iteration = 25
my_path = os.path.abspath(__file__ + "/../../")


class HeliumAtom(object):
    def __init__(self, a, t):
        self.alpha = a
        self.trial = t

    # Hamiltonian, non-interaction part
    @staticmethod
    def hamil_integrand(rr, a, b):
        return 4 * np.pi * (-2 + 3 * a * rr - 2 * a ** 2 * rr ** 3) \
               / rr * np.exp(-(a + b) * rr ** 2) * rr ** 2

    def hamiltonian(self, a, b):
        return quad(self.hamil_integrand, 0.0, np.inf, args=(a, b))[0]

    # Overlap intergral
    @staticmethod
    def overlap_integrand(rr, a, b):
        return 4 * np.pi * np.exp(-(a + b) * rr ** 2) * rr ** 2

    def overlap(self, a, b):
        return quad(self.overlap_integrand, 0.0, np.inf, args=(a, b))[0]

    # Hamiltonian, interaction part
    @staticmethod
    def hartree_integration(a, b, c, d):
        return 2 * np.pi ** (5 / 2) / (a + c) / (b + d) / np.sqrt(a + b + c + d)

    def hartree_matrix(self, alpha):
        tmp = np.array(
            [self.hartree_integration(aa, bb, cc, dd)
             for aa in alpha
             for cc in alpha
             for bb in alpha
             for dd in alpha]).reshape(4, 4, 4, 4)
        # The order of [aa, cc, bb, dd] is crucial
        return tmp

    @staticmethod
    def coefficient_matrix(coeff):
        return np.outer(coeff, coeff)

    def k_matrix(self, coeff):
        return np.tensordot(self.coefficient_matrix(coeff),
                            self.hartree_matrix(self.alpha), axes=2)

    # Solve eigenvalue problem
    def construct_eigval_problem(self, alpha):
        # Vectorize the table-like function
        vec_hamiltonian = np.vectorize(self.hamiltonian)
        h_ij = np.array([vec_hamiltonian(a, alpha) for a in alpha])

        # Vectorize the table-like function
        vec_overlap = np.vectorize(self.overlap)
        s_ij = np.array([vec_overlap(a, alpha) for a in alpha])

        return h_ij, s_ij

    def solve_eigenvalue_problem(self, alpha, coeff):
        h_ij, s_ij = self.construct_eigval_problem(alpha)
        k_ij = self.k_matrix(coeff)

        # Generalized eigenvalue Problem (H+K) x = E S x
        eigvals, eigvecs = eigh((h_ij + k_ij), s_ij, eigvals_only=False, type=1)
        return eigvals[0], eigvecs[:, 0]  # Only return ground state

    # Compute final result
    # Energy
    @staticmethod
    def hartree_energy(eigvec, k_matrix):
        return eigvec.dot(k_matrix).dot(eigvec)

    def total_energy(self, alpha, coeff):
        eigval = self.solve_eigenvalue_problem(alpha, coeff)[0]
        eigvec = coeff  # Eigvec is within the same step!
        return 2 * eigval - self.hartree_energy(eigvec, self.k_matrix(eigvec))

    def self_consistency_loop(self, iteration):
        # Be sure to make eigval and eigvec match!
        eigvals = [self.solve_eigenvalue_problem(self.alpha, self.trial)[0]]
        eigvecs = [self.trial]
        total_e = [self.total_energy(self.alpha, self.trial)]  # Total energy
        new_eigvec = self.solve_eigenvalue_problem(self.alpha, self.trial)[1]
        # while np.abs(
        #                 eigvals[-1] -
        #                 self.solve_eigenvalue_problem(self.alpha, new_eigvec)[0]
        # ) > 0.0000000001:
        #     eigvecs.append(new_eigvec)
        #     new_eigval = self.solve_eigenvalue_problem(self.alpha, new_eigvec)[0]
        #     eigvals.append(new_eigval)
        #     new_eigvec = self.solve_eigenvalue_problem(self.alpha, new_eigvec)[1]
        for i in range(iteration):
            eigvecs.append(new_eigvec)

            new_eigval = self.solve_eigenvalue_problem(self.alpha, new_eigvec)[0]
            eigvals.append(new_eigval)

            new_total_e = self.total_energy(self.alpha, new_eigvec)
            total_e.append(new_total_e)

            new_eigvec = self.solve_eigenvalue_problem(self.alpha, new_eigvec)[1]
        return eigvals, total_e, eigvecs

    # Density
    def density(self, rr, alpha, coeff):
        # This is not equal to overlap_integrand
        vec_overlap_integrand = np.vectorize(
            lambda rr, a, b: np.exp(-(a + b) * rr ** 2))
        phi_phi_mat = np.array(
            [vec_overlap_integrand(rr, a, alpha) for a in alpha])
        coeff_mat = self.coefficient_matrix(coeff)
        # Elementwise product, then sum all
        return np.multiply(coeff_mat, phi_phi_mat).sum()


he = HeliumAtom(alpha, trial)
eigvals, total_e, eigvecs = he.self_consistency_loop(iteration)

# Plot
# Total energy as a function of interactions
plt.style.use('classic')


def my_plot_1():
    iter_list = list(range(1 + iteration))
    fig1 = plt.figure()
    plt.plot(iter_list, total_e)
    plt.ylim((-2.89, -2.85))
    ml = MultipleLocator(1)
    plt.axes().xaxis.set_minor_locator(ml)
    plt.xlabel(r'$n$', fontsize=16)
    plt.ylabel(r'$\varepsilon$ (Hartree)', fontsize=16)
    # plt.show()
    fig1.savefig(my_path + "/images/pro_2_b_1.pdf")


# Final density
def my_plot_2():
    r = np.linspace(0, 3, endpoint=True, num=500)
    fig2 = plt.figure()
    plt.plot(r, [he.density(rr, alpha, eigvecs[-1]) for rr in r],
             label=r'interaction value')
    plt.plot(r, [mod.density(rr, alpha) for rr in r],
             label=r'non-interaction value')
    plt.xlabel(r'$r$ (Bohr)', fontsize=16)
    plt.ylabel(r'Density $|\psi(r)|^2$', fontsize=16)
    plt.legend(loc="best")
    # plt.show()
    fig2.savefig(my_path + "/images/pro_2_b_2.pdf")


# Write to Excel
def write_to_excel():
    np.savetxt(my_path + "/code/2-b-eigvecs.csv", eigvecs, delimiter=',')
    workbook = xlsxwriter.Workbook(my_path + "/code/2_b.xlsx")
    worksheet = workbook.add_worksheet()
    row = 0
    worksheet.write_string(row, 0, "eigenvalue")
    worksheet.write_string(row, 1, "total energy")
    for col, data in enumerate([eigvals, total_e]):
        worksheet.write_column(row + 1, col, data)

    workbook.close()

# my_plot_1()
# my_plot_2()
# write_to_excel()
