# %% imports
import numpy as np

import matplotlib.pyplot as plt
import SHELL4 as fems4
import SOLVE as solv
import MESH as msh

from joblib import Parallel, delayed
import pandas as pd
import matplotlib
from numpy.linalg import norm
matplotlib.use('qtagg')

# %%
#  

if __name__ == "__main__":
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('corcos_vel.csv')
    # Accessing columns
    f_0 = df['f0'].values
    f_1 = df['f1'].values
    f_2 = df['f2'].values
    Gvv_0 = df['Gvv0'].values
    Gvv_1 = df['Gvv1'].values
    Gvv_2 = df['Gvv2'].values
    #%%
    # Propriedades
    E = 200e9
    rho = 7850
    v = 0.3
    alpha_x = 0.11
    alpha_y = 0.7
    rho_air =  1.1845
    v_air = 1.8444e-5
    d_ = 0.0024 
    # Proportinal damping
    alpha = 0
    beta = 1e-6
    # Espessura
    h_init = 0.0159
    eta = 0.05
    U0 = 178.8
    Re_d = 8 * U0*d_/v_air
    tau_w = (0.0225 * rho_air * U0**2)/(Re_d**0.25)
    c0 = 343
    #%%
    #################### MALHA QUAD4 RETANGULO ##################################
    lx = 0.47 # Comprimento
    ly = 0.37 # Altura

    tamanho_elemento = 0.02
    coord, connect, nodes_faceX1, nodes_faceX2, nodes_faceY1, nodes_faceY2, nodes_middle = msh.malha2D_QUAD4(lx,ly,tamanho_elemento)
    nnode = len(coord)
    nel = len(connect)
    h = h_init*np.ones((nel, 1))
    ########### Condição de apoio nas bordas ####################################
    nodes_bordaX = nodes_faceX1
    #nodes_bordaX = np.append(nodes_faceX1, nodes_faceX2)
    nodes_bordaY = np.append(nodes_faceY1, nodes_faceY2)
    
    nodes_borda = np.append(nodes_bordaX, nodes_bordaY)
    nnode_borda = len(nodes_borda)
    mid_dofs = []
    nnode_middle = len(nodes_middle)
    for i in range(nnode_middle):
        mid_dofs_i = np.array([2]) + 5*nodes_borda[i]
        mid_dofs = np.append(mid_dofs, mid_dofs_i)

    fixed_dofs = []
    for i in range(nnode_borda):
        fixed_dofs_i = np.array([0,1,2,3,4]) + 5*nodes_borda[i]
        fixed_dofs = np.append(fixed_dofs, fixed_dofs_i)
    all_dofs = np.arange(5*nnode)
    z_dofs = np.arange(2,5*nnode,5)
    fixed_dofs = fixed_dofs.astype(int)
    free_dofs = np.delete(all_dofs,fixed_dofs)
    xy = -ly/2 + 0.12
    xx = -lx/2 + 0.15
    size = tamanho_elemento
    
    check_node = np.where(
        (coord[:, 1] >= (xy)-(size/2)) & (coord[:, 1] <= (xy)+(size/2)) & (coord[:, 0] >= (xx)-(size/2)) & (coord[:, 0] <= (xx)+(size/2) )  # Dentro do intervalo y
    )[0]
    check_dof = check_node*5-2
    # %%
    ###################### ASSEMBLY #################################
    ind_rows, ind_cols = fems4.generate_ind_rows_cols(connect)
    stif_matrix, mass_matrix = fems4.stif_mass_matrices(coord, connect, nnode, nel, ind_rows, ind_cols, E, v, rho, h)
    
    ################### ANÁLISE MODAL ###############################
    modes = 30
    modal_shape = np.zeros((5*nnode,modes))
    natural_frequencies, modal_shape[free_dofs,:] = solv.modal_analysis(stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], modes, which='LM', sigma=0.01)
    print('Frequências Naturais:')
    print(natural_frequencies)
    # %%
    freq = np.linspace(50,2000,200)
    Vr = modal_shape[z_dofs,:]  # Modos normais
    wn = 2 * np.pi * natural_frequencies  # Frequências naturais (rad/s)
    glf = check_node[0]  # Grau de liberdade da força
    w = 2 * np.pi * freq  # Frequências angulares

    # Inicializando as FRFs    
    Hv = np.zeros((len(z_dofs), len(freq)), dtype=complex)
    
    # Pré-calculando termos repetidos para otimização
    eta_wn2 = eta * wn**2  # Termo de amortecimento modal
    wn2 = wn**2  # Frequência natural ao quadrado

    # Vetorização dos cálculos das FRFs
    for glr in range(len(z_dofs)):
        # Pré-calculando para cada grau de liberdade de resposta
        for k in range(modes):
            # Cálculo vetorizado para todas as frequências de uma vez
            den = (1j * w)/ (wn2[k] - w**2 + 1j * eta_wn2[k])
            # FRF pontual (mesmo ponto de força)
            
            Hv[glr, :] += Vr[glr, k] * Vr[glf, k] * den

    #%% 
    # Função paralelizada que será aplicada em cada frequência
    def compute_Gamma_TBL(kk, w, Uc, ksix, ksiy, alpha_x, alpha_y, Aij):
        ksix_Uc = np.abs(w * ksix / Uc)
        ksiy_Uc = np.abs(w * ksiy / Uc)
        # Precomputando o termo Gamma
        exp_x = np.exp(-alpha_x * ksix_Uc)  # Parte exponencial para ksix
        exp_y = np.exp(-alpha_y * ksiy_Uc)  # Parte exponencial para ksiy
        complex_exp = np.exp(1j * w * ksix / Uc)  # Parte exponencial complexa

        Gamma = (1 + alpha_x * ksix_Uc) * exp_x * complex_exp * exp_y
        return Gamma, kk
    
    # Pré-computando constantes fora dos loops
    w_array = 2 * np.pi * freq  # Frequências angulares
    Uc_array = U0 * (0.59 + 0.30 * np.exp(-0.89 * w_array * d_ / U0))  # Precomputando Uc para todas as frequências
    phi_pp_TBL = np.array((tau_w**2 * d_ / U0) * (5.1 / (1 + 0.44 * (w_array * d_ / U0)**(7/3))), ndmin=1)

    # Diferenças de coordenadas
    ksix = coord[:, None, 0] - coord[None, :, 0]  # Diferenças em x
    ksiy = coord[:, None, 1] - coord[None, :, 1]  # Diferenças em y
    Aij = (tamanho_elemento)**2  # Precomputando Aij fora do loop
    Aij = 0.47*0.37/nel  # Precomputando Aij fora do loop

    # Usando joblib para paralelizar o loop sobre frequências
    results_TBL = Parallel(n_jobs=-2)(delayed(compute_Gamma_TBL)(
        kk, w, Uc, ksix, ksiy, alpha_x, alpha_y, Aij) for kk, (w,  Uc) in enumerate(zip(w_array, Uc_array)))

    # Atualizando Gamma_w com os resultados paralelizados
    Gamma_TBL = np.zeros((nnode,nnode,len(freq)), dtype=complex)   
    for Gamma, kk in results_TBL:
        Gamma_TBL[:,:,kk] = Gamma

    # %%
    # Function to compute phi_pb for a specific frequency index kk
    def compute_Gamma_DAF(kk, w, coord, c0, nnode):
        k0 = w / c0
        Gamma = np.zeros((nnode, nnode), dtype=complex)  # For storing results of phi_pb for this frequency
        for ii, x1 in enumerate(coord):
            for jj, x2 in enumerate(coord):
                if ii != jj:
                    distance = norm(abs(x1 - x2))
                    Gamma[ii, jj] = np.sin(k0 * distance) / (k0 * distance)
                else:
                    Gamma[ii, jj] = 1  # Evitar divisão por zero
        return Gamma, kk

    # Preallocate arrays
    Gamma_DAF = np.zeros((nnode, nnode, len(w_array)), dtype=complex)

    # Parallel computation of phi_pb using joblib
    results_DAF = Parallel(n_jobs=-1)(delayed(compute_Gamma_DAF)(
        kk, w, coord, c0, nnode) for kk, w in enumerate(w_array))

    # Combine the results back into phi_pb
    for Gamma, kk in results_DAF:
        Gamma_DAF[:, :, kk] = Gamma
    #%% 
    Gvv_TBL = np.zeros(len(freq), dtype=complex)
    Gvv_DAF = np.zeros(len(freq), dtype=complex)
        
    for ii in range(len(freq)):
        Gvv_TBL[ii] = np.conj(Hv[:,ii].T)@(phi_pp_TBL[ii]*Gamma_TBL[:,:,ii])@Hv[:,ii]
        Gvv_DAF[ii] = np.conj(Hv[:,ii].T)@(phi_pp_TBL[ii]*Gamma_DAF[:,:,ii])@Hv[:,ii]
        
    # %%

    plt.semilogy(freq,(np.abs(Gvv_TBL)))
    plt.semilogy(freq,(np.abs(Gvv_DAF)))

    
    #plt.semilogy(f_0, (np.abs(Gvv_0)))
    #plt.semilogy(f_1, (np.abs(Gvv_1)))
    plt.semilogy(f_2, (np.abs(Gvv_2)))
    
    #plt.plot(freq, np.log10(abs((Guu))))
    #plt.plot(freq, np.log10(abs((Gxx))))
    #plt.semilogy(freq, abs(Guu))
    plt.grid(True, which='major')
    #plt.ylim(1e-19, 1e-9)
    #plt.xlim(0,600)

    plt.show()    
# %%
