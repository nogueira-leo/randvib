from sympy import symbols, diff, Matrix, simplify, solve, pi, I
import numpy as np

# Constants
E = 7.1e10  # Young's modulus
nu = 0.33  # Poisson's ratio
Iz = 0.002  # Moment of inertia
rho = 2700  # Density
h = 0.002  # Element thickness
L1 = 0.7  # Length of the beam in x-direction
L2 = 0.5  # Length of the beam in y-direction


# Symbolic variables
qsi, eta = symbols('qsi eta')

# Shape functions
N1 = (1 - qsi) / 2
N2 = (1 + qsi) / 2
M1 = (1 - eta) / 2
M2 = (1 + eta) / 2

# Derivatives of shape functions
dN1_dqsi = diff(N1, qsi)
dN2_dqsi = diff(N2, qsi)
dM1_deta = diff(M1, eta)
dM2_deta = diff(M2, eta)


# Element stiffness matrix
B = Matrix([[dN1_dqsi * M1, dN1_dqsi * M2, dN2_dqsi * M1, dN2_dqsi * M2],
             [N1 * dM1_deta, N1 * dM2_deta, N2 * dM1_deta, N2 * dM2_deta]])

D = E * h * Matrix([[1 - nu, nu, 0],
                     [nu, 1 - nu, 0],
                     [0, 0, (1 - 2 * nu) / 2]])

B_e = simplify(B.T @ D @ B  h)

# Element mass matrix
A = pho * h * Iz * Matrix([[N1 * M1, N1 * M2, N2 * M1, N2 * M2]])

M_e = simplify(A.T * A * h)

# Global stiffness and mass matrices
K = np.zeros((2 * (nL1 + 1) * (nL2 + 1), 2 * (nL1 + 1) * (nL2 + 1)))
M = np.zeros((2 * (nL1 + 1) * (nL2 + 1), 2 * (nL1 + 1) * (nL2 + 1)))

for j in range(nL1):
    for k in range(nL2):
        # Calculate element stiffness and mass matrices
        B_e_val = B_e.subs([(qsi, -1 + 2 * j / nL1), (eta, -1 + 2 * k / nL2)])
        M_e_val = M_e.subs([(qsi, -1 + 2 * j / nL1), (eta, -1 + 2 * k / nL2)])

        # Assemble global matrices
        K[2 * j:2 * (j + 1), 2 * k:2 * (k + 1)] += B_e_val
        M[2 * j:2 * (j + 1), 2 * k:2 * (k + 1)] += M_e_val

# Apply boundary conditions
K_reduced = K[2:, 2:]
M_reduced = M[2:, 2:]

# Solve generalized eigenvalue problem
w2, V = np.linalg.eig(np.linalg.inv(M_reduced) @ K_reduced)

# Natural frequencies
w = np.sqrt(w2)

# Points of excitation and response
x = np.array([0.14, 0.08])
xf = np.array([0.44, 0.04])

# Frequency response function
Rp_m = np.zeros(len(w), dtype=complex)
Rt_m = np.zeros(len(w), dtype=complex)

for i in range(len(w)):
    # Calculate mode shape at the point of excitation
    V_x = V[2 * (x[0] * nL1 / L1) + 1, i]
    V_xf = V[2 * (xf[0] * nL1 / L1) + 1, i]

    # Calculate frequency response function
    Rp_m[i] = V_xf * V_x / (w[i] ** 2 - (2 * pi * freq) ** 2 + I * eta_p * (2 * pi * freq) ** 2)
    Rt_m[i] = V_xf * V_x / (w[i] ** 2 - (2 * pi * freq) ** 2 + I * eta_p * (2 * pi * freq) ** 2)

    # Plot frequency response functions
    import matplotlib.pyplot as plt
    
    plt.figure(1)
    plt.semilogy(freq, np.abs(Rp_m))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Receptance Modulus')
    plt.title('Receptance Modulus vs Frequency')
    
    plt.figure(2)
    plt.semilogy(freq, np.abs(Rt_m))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Transfer Modulus')
    plt.title('Transfer Modulus vs Frequency')
    
    plt.show()