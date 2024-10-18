# %%
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# %%
# Parâmetros da placa
Lx = 0.47  # Comprimento da placa (m)
Ly = 0.37  # Largura da placa (m)
thickness = 0.01  # Espessura da placa (m)
E = 2.1e11  # Módulo de Young (Pa)
rho = 7800  # Densidade da placa (kg/m^3)
nu = 0.3  # Coeficiente de Poisson


#%%
# Propriedades da placa
L1 = 0.47  # Comprimento da placa (m)
L2 = 0.37  # Largura da placa (m)
h = 0.01  # Espessura da placa (m)
Iz = h**3 / 12
pho = 7800  # Densidade da placa (kg/m^3)
E = 2.1e11  # Módulo de Young (Pa)
nu = 0.3  # Coeficiente de Poisson
eta_p = 0.01

# Faixa de frequência
freq = np.arange(20, 301, 1)

# Montando matriz do material
E1 = E / (1 - nu**2)
G = E / (2 * (1 + nu))
D = np.array([[E1, E1 * nu, 0], [E1 * nu, E1, 0], [0, 0, G]])

# Tamanho do elemento desejado
L = 0.02

# Definindo a malha
nL1 = int(np.ceil(L1 / L))
nL2 = int(np.ceil(L2 / L))
Le1 = L1 / nL1
Le2 = L2 / nL2

# Criando os nós
count = 0
node = np.zeros((nL1 + 1, nL2 + 1), dtype=int)
for j in range(nL1 + 1):
    for k in range(nL2 + 1):
        count += 1
        node[j, k] = count

Nn = count  # Número de nós na malha
Ngl = Nn * 3  # Número de graus de liberdade

# Definir pontos de resposta e aplicação de força
x = np.array([0.14, 0.08])
xf = np.array([0.44, 0.04])
node_r = node[int(x[0] / Le1), int(x[1] / Le2)]
node_f = node[int(xf[0] / Le1), int(xf[1] / Le2)]

# Definindo os elementos
count = 0
elem = np.zeros((nL1, nL2), dtype=int)
malha = []
for j in range(nL1):
    for k in range(nL2):
        count += 1
        elem[j, k] = count
        malha.append([count, node[j, k], node[j + 1, k], node[j + 1, k + 1], node[j, k + 1]])

Ne = count  # Número de elementos na malha

# Determinando simbolicamente as matrizes do elemento
def montar_matrizes_elemento(Le1, Le2, pho, h, Iz, D):
    # Função para montar as matrizes de massa (Me) e rigidez (Ke) do elemento
    a = Le1 / 2
    b = Le2 / 2
       
    
    # Define symbolic variables
    qsi, eta = sp.symbols('qsi eta')

    # Define shape function
    p = sp.Matrix([1, qsi, eta, qsi**2, qsi*eta, eta**2, qsi**3, qsi**2*eta, qsi*eta**2, eta**3, qsi**3*eta, qsi*eta**3])

    # Define derivatives of shape function
    dp_dqsi = sp.diff(p, qsi)
    dp_deta = sp.diff(p, eta)

    # Montando matriz de tranformacao de coef. para coord. locais
    
    nlc = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

    # Initialize element transformation matrix
    Ae = np.zeros((12, 12))

    for j in range(4):
        ps = p.subs({qsi: nlc[j, 0], eta: nlc[j, 1]})
        dp_dqsi_s = dp_dqsi.subs({qsi: nlc[j, 0], eta: nlc[j, 1]})
        dp_deta_s = dp_deta.subs({qsi: nlc[j, 0], eta: nlc[j, 1]})
        Ae[3*j,:]=ps.T
        Ae[3*j+1,:]=(dp_dqsi_s/b).T
        Ae[3*j+2,:]=(dp_deta_s/a).T
        #Ae[3*j:3*j-2+3, :] = np.array([ps, dp_dqsi_s/b, dp_deta_s/a]).reshape(-1, 1)

    # Inverse of element transformation matrix
    Ae_inv = np.linalg.inv(Ae)
    
    # Mass matrix of the element
    int_m_qsi_eta = pho * h * p @ p.T * a * b
    int_m_eta = sp.integrate(int_m_qsi_eta, (qsi, -1, 1))
    int_m = sp.integrate(int_m_eta, (eta, -1, 1))
    Me = np.array(Ae_inv.T @ int_m @ Ae_inv)
    
    # Stiffness matrix of the element
    P = sp.Matrix(np.zeros((3, 12)))
    P[0, :] = 1 / a**2 * sp.diff(p, qsi, 2).T
    P[1, :] = 1 / b**2 * sp.diff(p, eta, 2).T
    P[2, :] = 2 / (a * b) * sp.diff(dp_dqsi, eta).T
    
    int_k_qsi_eta = Iz * P.T @ D @ P * a * b
    int_k_eta = sp.integrate(int_k_qsi_eta, (qsi, -1, 1))
    int_k = sp.integrate(int_k_eta, (eta, -1, 1))
    Ke = np.array(Ae_inv.T @ int_k @ Ae_inv)
    # Aqui você pode implementar a montagem de Ae_inv, Me e Ke
    # Esse código foi omitido por simplicidade, mas deve seguir o mesmo princípio
    # do código MATLAB original

    # Retornar as matrizes de massa e rigidez do elemento
    return Me, Ke

# Montando as matrizes de massa e rigidez totais
M = sp.Matrix(np.zeros((Nn * 3, Nn * 3)))
K = sp.Matrix(np.zeros((Nn * 3, Nn * 3)))

for j in range(Ne):
    ele = malha[j]
    Me, Ke = montar_matrizes_elemento(Le1, Le2, pho, h, Iz, D)
    for k in range(4):
        for m in range(4):
            kk = ele[k + 1]
            mm = ele[m + 1]
            M[kk * 3:kk * 3+2, mm * 3:mm * 3+2] += Me[k * 3:k * 3+2, m * 3:m * 3+2]
            K[kk * 3:kk * 3+2, mm * 3:mm * 3+2] += Ke[k * 3:k * 3+2, m * 3:m * 3+2]

# Aplicando as condições de contorno
node_cc = np.concatenate([node[0, :], node[-1, :], node[1:-1, 0], node[1:-1, -1]])
gl_cc = np.sort(np.concatenate([3 * node_cc - 3, 3 * node_cc - 2, 3 * node_cc - 1]))

M = np.delete(np.delete(M, gl_cc, axis=0), gl_cc, axis=1)
K = np.delete(np.delete(K, gl_cc, axis=0), gl_cc, axis=1)

# Solução modal
n_modes = 50
eigvals, Vr = eigh(K, M)
fn = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
wn = 2 * np.pi * fn

# Pontos de excitação e resposta
glr = node_r * 3 - 2
glf = node_f * 3 - 2

# Calculando as FRFs
Rp_m = np.zeros(len(freq), dtype=complex)
Rt_m = np.zeros(len(freq), dtype=complex)
w = 2 * np.pi * freq

for j in range(len(freq)):
    for k in range(n_modes):
        Rp_m[j] += Vr[glf, k] * Vr[glf, k] / (wn[k]**2 - w[j]**2 + 1j * eta_p * wn[k]**2)
        Rt_m[j] += Vr[glr, k] * Vr[glf, k] / (wn[k]**2 - w[j]**2 + 1j * eta_p * wn[k]**2)

# Comparação da solução modal
plt.figure(1)
plt.semilogy(freq, np.abs(Rp_m))
plt.title("Receptância Modal Pontual")

plt.figure(2)
plt.semilogy(freq, np.abs(Rt_m))
plt.title("Receptância Modal de Transferência")

plt.show()



#%%
# Frequência de excitação
frequencies = np.linspace(0, 1000, 500)  # Faixa de frequências em Hz

# Modelo de Corcos - correlação espacial da pressão turbulenta
def corcos_model(Uc, f, L1, L2, alpha1=0.1, alpha2=0.3):
    """
    Uc: velocidade convectiva (m/s)
    f: frequência (Hz)
    L1: direção do fluxo
    L2: direção perpendicular ao fluxo
    alpha1 e alpha2: parâmetros de decaimento de Corcos
    """
    omega = 2 * np.pi * f
    return np.exp(-alpha1 * np.abs(omega * L1 / Uc)) * np.exp(-alpha2 * np.abs(omega * L2 / Uc))

# Exemplo de aplicação
Uc = 44.7  # Velocidade do fluxo em m/s
L1, L2 = np.meshgrid(np.linspace(-Lx / 2, Lx / 2, 50), np.linspace(-Ly / 2, Ly / 2, 50))

# Frequência de 500 Hz
corcos_pressure = corcos_model(Uc, 500, L1, L2)

def calculate_plate_response(pressure, K, M):
    """
    pressure: pressão aplicada
    K: matriz de rigidez
    M: matriz de massa
    """
    # Resolvendo a equação de movimento da placa (simplificado)
    frequencies, modes = np.linalg.eig(K, M)
    response = np.dot(modes.T, pressure)
    return response

def diffuse_field_excitation(frequency, Lx, Ly, c0=343):
    """
    frequency: frequência da excitação (Hz)
    Lx, Ly: dimensões da placa (m)
    c0: velocidade do som no ar (m/s)
    """
    k0 = 2 * np.pi * frequency / c0  # Número de onda
    angles = np.linspace(0, 2 * np.pi, 100)  # Distribuição de ângulos
    pressure = np.zeros((len(angles), len(angles)))

    for i, theta in enumerate(angles):
        for j, phi in enumerate(angles):
            pressure[i, j] = np.cos(k0 * Lx * np.cos(theta)) * np.sin(k0 * Ly * np.sin(phi))

    return np.sum(pressure)

# Aplicando para uma frequência de 500 Hz
diffuse_pressure = diffuse_field_excitation(500, Lx, Ly)


import matplotlib.pyplot as plt

# Pressão aplicada pela TBL e DAF
response_TBL = calculate_plate_response(corcos_pressure, K, M)
response_DAF = calculate_plate_response(diffuse_pressure, K, M)

# Comparação das respostas
plt.plot(frequencies, response_TBL, label="TBL")
plt.plot(frequencies, response_DAF, label="DAF")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Resposta")
plt.legend()
plt.show()
