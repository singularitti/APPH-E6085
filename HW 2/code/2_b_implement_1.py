#!/usr/bin/env python3
"""Problem 2(b).
Self-consistent iteration.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad, nquad

alpha = np.array([0.298073, 1.242567, 5.782948, 38.474970])
trial = np.array([0.08704173, 0.52509737, 0.51920929, 0.31233737])


class EigenvalueProblem(object):
    def __init__(self, a, t):
        self.alpha = a
        self.trial = t

    # Hamiltonian, non-interaction part
    @staticmethod
    def hamil_integrand(rr, a, b):
        return 4 * np.pi * (-2 + 3 * a * rr - 2 * a ** 2 * rr ** 3) / rr * np.exp(-(a + b) * rr ** 2) * rr ** 2

    def hamiltonian(self, a, b):
        return quad(self.hamil_integrand, 0.0, np.inf, args=(a, b))[0]

    # Overlap intergral
    @staticmethod
    def overlap_integrand(rr, a, b):
        return 4 * np.pi * np.exp(-(a + b) * rr ** 2) * rr ** 2

    def overlap(self, a, b):
        return quad(self.overlap_integrand, 0.0, np.inf, args=(a, b))[0]

    # Hamiltonian, interaction part
    def hartree_integrand(self, coeff):
        def hartree_potential(rr2):
            hermgauss(2)
            return np.array([coeff[ii] * coeff[jj] *
                             np.exp(-(self.alpha[ii] +
                                      self.alpha[jj]) * rr2 ** 2)
                             for ii in range(0, 4) for jj in range(0, 4)]).sum()

        def length(theta, rr1, rr2):
            return 1 / np.sqrt(rr1 ** 2 + rr2 ** 2 -
                               2 * rr1 * rr2 * np.cos(theta))

        def tmp(theta, rr1, rr2):
            return 8 * np.pi ** 2 * rr1 ** 2 * rr2 ** 2 * \
                   np.sin(theta) * hartree_potential(rr2) * \
                   length(theta, rr1, rr2)

        def integrand(ii, jj, theta, rr1, rr2):
            return np.exp(-(self.alpha[ii] + self.alpha[jj]) * rr1 ** 2) * tmp(theta, rr1, rr2)

        return [
            nquad(lambda theta, rr1, rr2: integrand(i, j, theta, rr1, rr2),
                  [[0, np.pi], [0, np.inf], [0, np.inf]]) for i in range(0, 4) for j in range(0, 4)]


hat = EigenvalueProblem(alpha, trial)
print(hat.hartree_integrand(trial))
# I can't finish it because it takes long time to calculate :)
