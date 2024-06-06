#!/usr/bin/env python
"""Problem 1(c).
Compared the ground state eigenfunctions obtained using different numbers of basis functions with the exact solution.
"""

import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np

from hw2.ex1 import exact_solution, wavefunction

matplotlib.style.use("classic")
plt.rcParams.update(
    {
        "axes.titlesize": 12,  # Font size for axes title
        "axes.labelsize": 10,  # Font size for x and y labels
        "xtick.labelsize": 8,  # Font size for x tick labels
        "ytick.labelsize": 8,  # Font size for y tick labels
        "legend.fontsize": 10,  # Font size for legend
        "figure.titlesize": 12,  # Font size for figure title
    }
)

if __name__ == "__main__":
    r = np.linspace(0, 3, endpoint=True, num=500)
    alpha_1 = np.asfarray([13])
    alpha_2 = np.asfarray([13, 1.96])
    alpha_3 = np.asfarray([13, 1.96, 0.44])
    alpha_4 = np.asfarray([13, 1.96, 0.44, 0.12])
    alphas = [alpha_1, alpha_2, alpha_3, alpha_4]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotting and filling the area for each wavefunction
    for i, alpha in enumerate(alphas, start=1):
        (line,) = ax.plot(r, list(map(wavefunction(alpha, 0), r)), label=rf"$n={i}$")
        ax.fill_between(
            r, list(map(wavefunction(alpha, 0), r)), 0, alpha=0.3, color=line.get_color()
        )
    # Plotting and filling the area for the exact wavefunction
    (line_exact,) = ax.plot(
        r, np.fromiter(map(exact_solution, r), dtype=float), label=r"exact wavefunction"
    )
    ax.fill_between(
        r,
        np.fromiter(map(exact_solution, r), dtype=float),
        0,
        alpha=0.3,
        color=line_exact.get_color(),
    )
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlabel(r"$r$ (Bohr)")
    ax.set_ylabel(r"wavefunction")
    ax.legend(loc="best")
    ax.set_title(
        "Ground states obtained using different # of basis functions vs. the exact solution"
    )
    fig.savefig("1_c.pdf")
