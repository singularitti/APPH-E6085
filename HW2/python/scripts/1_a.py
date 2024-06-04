#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from hw2.ex1 import *

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
