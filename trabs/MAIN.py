# %% imports
import numpy as np
import scipy
from scipy.fft import fftn, fft
import matplotlib.pyplot as plt
import SHELL4 as fems4
import SOLVE as solv
import MESH as msh
from VTK_FUNCS import vtk_write_displacement, vtk_write_modal, vtk_write_velocity
# MMA
#from __future__ import division
from joblib import Parallel, delayed
from typing import Tuple
import os
#%%
if __name__ == "__main__":
    
    ############# ATENÇÃO!!! EXEMPLO ADAPTADO PARA O CASO ESTÁTICO!!! #################
    ############### MATERIAL, AMORTECIMENTO E ESPESSURA GLOBAL (SI) ####################
    E = 200e9
    rho = 7850
    v = 0.3
    alpha_x = 0.11
    alpha_y = 0.70
    rho_air = 1.18
    v_air = 1.48e-5
    d_ = 0.0024
    # Proportinal damping
    alpha = 0
    beta = 1e-6
    # Espessura
    h_init = 0.0159
    eta = 0.05
    U0 = 89.7
    Re_d = 8 * U0*d_/v_air
    tau_w = 0.0225 * rho_air * U0**2/Re_d
    D = E * h_init**3/(12*(1-v**2))
    c0 = 343
    #%%
    #################### MALHA QUAD4 RETANGULO ##################################
    lx = 0.47 # Comprimento
    ly = 0.37 # Altura

    tamanho_elemento = 0.01
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
    modes = 50
    modal_shape = np.zeros((5*nnode,modes))
    natural_frequencies, modal_shape[free_dofs,:] = solv.modal_analysis(stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], modes, which='LM', sigma=0.01)
    print('Frequências Naturais:')
    print(natural_frequencies)
    # %%
    freq = np.linspace(10,2000,200)
    Vr = modal_shape[z_dofs,:]  # Modos normais
    wn = 2 * np.pi * natural_frequencies  # Frequências naturais (rad/s)
    glf = int(check_node)  # Grau de liberdade da força
    w = 2 * np.pi * freq  # Frequências angulares

    # Inicializando as FRFs
    Rp_m = np.zeros(len(freq), dtype=complex)
    Hu = np.zeros((len(z_dofs), len(freq)), dtype=complex)
    Hv = np.zeros((len(z_dofs), len(freq)), dtype=complex)
    
    # Pré-calculando termos repetidos para otimização
    eta_wn2 = eta * wn**2  # Termo de amortecimento modal
    wn2 = wn**2  # Frequência natural ao quadrado

    # Vetorização dos cálculos das FRFs
    for glr in range(len(z_dofs)):
        # Pré-calculando para cada grau de liberdade de resposta
        for k in range(modes):
            # Cálculo vetorizado para todas as frequências de uma vez
            den = (1)/ (wn2[k] - w**2 + 1j * eta_wn2[k])
                        
            # FRF pontual (mesmo ponto de força)
            Rp_m += (Vr[glf, k]**2) * den
            
            # FRF de transferência (resposta em outro ponto)
            Hu[glr, :] += Vr[glr, k] * Vr[glf, k] * den
            Hv[glr, :] += Vr[glr, k] * Vr[glf, k] * den * 1j * w

    # %%
    # Função paralelizada que será aplicada em cada frequência
    def compute_force(kk, w, phi_pp, Uc, ksix, ksiy, alpha_x, alpha_y, Aij):
        ksix_Uc = np.abs(w * ksix / Uc)
        ksiy_Uc = np.abs(w * ksiy / Uc)

        # Precomputando o termo Gamma
        exp_x = np.exp(-alpha_x * ksix_Uc)  # Parte exponencial para ksix
        exp_y = np.exp(-alpha_y * ksiy_Uc)  # Parte exponencial para ksiy
        complex_exp = np.exp(1j * w * ksix / Uc)  # Parte exponencial complexa

        Gamma = (1 + alpha_x * ksix_Uc) * exp_x * complex_exp * exp_y
        Gxx = (Aij * phi_pp * Gamma * Aij)
        ## TODO: export force/pressure instead of Gamma
        # Retornar o valor de Gamma para essa frequência
        return Gamma, kk
    
    # Pré-computando constantes fora dos loops
    f = freq
    Gf = np.zeros((nnode, f.size), dtype=np.complex64)
    Gamma_w = np.zeros((nnode, nnode, f.size), dtype=np.complex64)

    w_array = 2 * np.pi * f  # Frequências angulares
    Uc_array = U0 * (0.59 + 0.30 * np.exp(-0.89 * w_array * d_ / U0))  # Precomputando Uc para todas as frequências
    phi_pp_array = np.array((tau_w**2 * d_ / U0) * (5.1 / (1 + 0.44 * (w_array * d_ / U0)**(7/3))), ndmin=1)

    # Diferenças de coordenadas
    ksix = coord[:, None, 0] - coord[None, :, 0]  # Diferenças em x
    ksiy = coord[:, None, 1] - coord[None, :, 1]  # Diferenças em y
    Aij = (tamanho_elemento / 2)**2  # Precomputando Aij fora do loop

    # Usando joblib para paralelizar o loop sobre frequências
    results = Parallel(n_jobs=-1)(delayed(compute_force)(
        kk, w, phi_pp, Uc, ksix, ksiy, alpha_x, alpha_y, Aij) for kk, (w, phi_pp, Uc) in enumerate(zip(w_array, phi_pp_array, Uc_array)))

    # Atualizando Gamma_w com os resultados paralelizados
    for Gamma, kk in results:
        Gamma_w[:, :, kk] = Gamma
    #%%
    H1 = np.ones((nnode,1))


    Guu = np.zeros(len(freq), dtype=complex)   
    Gvv = np.zeros(len(freq), dtype=complex)   
       
    for ii in range(len(freq)):

        Guu[ii] = np.conj(Hu[:,ii]).T@(phi_pp_array[ii]*Gamma_w[:,:,ii])@Hu[:,ii]
        Gvv[ii] = np.conj(Hv[:,ii]).T@(Aij*phi_pp_array[ii]*Gamma_w[:,:,ii]*Aij)@Hv[:,ii]
        
    #%%
    plt.plot(freq, np.log10(abs(np.sqrt(Gvv))))
    #plt.semilogy(freq, abs(Guu))
    plt.grid(True)
    plt.ylim(-20, -10)
    #plt.xlim(0,600)
    
    # %%    
