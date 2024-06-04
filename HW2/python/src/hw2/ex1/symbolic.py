#!/usr/bin/env python
import sympy as sp

# Define variables and symbols
r = sp.symbols("r", real=True, positive=True)
alpha_i, alpha_j = sp.symbols("alpha_i alpha_j", real=True, positive=True)


# Function to define the Gaussian basis functions
def phi(alpha):
    return sp.exp(-alpha * r**2)


# Function to calculate the integrand
def integrand(phi_i, phi_j):
    # Define the Laplacian operator in spherical coordinates, you cannot split it out
    laplacian_phi_j = (1 / r) * sp.diff(sp.diff(r * phi_j, r), r)
    # Define the Hamiltonian operator
    H_phi_j = -laplacian_phi_j / 2 - phi_j / r
    # Define the integration measure
    measure = 4 * sp.pi * r**2
    integrand_expr = phi_i.conjugate() * H_phi_j * measure
    return sp.simplify(integrand_expr)


# Function to calculate the braket
def braket(phi_i, phi_j):
    integrand_expr = integrand(phi_i, phi_j)
    return sp.simplify(sp.integrate(integrand_expr, (r, 0, sp.oo)))


if __name__ == "__main__":
    # %%
    # Define the Gaussian basis functions
    phi_i = phi(alpha_i)
    phi_j = phi(-alpha_j)
    # Calculate and display the braket value symbolically
    integrand_expr = integrand(phi_i, phi_j)
    braket_value = braket(phi_i, phi_j)
    # %%
    # Define specific numeric values for alpha_i and alpha_j
    alpha_values = [13.0, 1.96, 0.44, 0.12]
    # Generate the matrix of braket values
    hamiltonian = sp.N(
        sp.Matrix(
            [
                [braket(phi(alpha_i), phi(alpha_j)) for alpha_j in alpha_values]
                for alpha_i in alpha_values
            ]
        )
    )
