# %% imports
import numpy as np
import scipy
import matplotlib.pyplot as plt
import SHELL4 as fems4
import SOLVE as solv
import MESH as msh
from VTK_FUNCS import vtk_write_displacement, vtk_write_modal, vtk_write_velocity
# MMA
#from __future__ import division
from mmapy import mmasub, kktcheck
from util import setup_logger
from typing import Tuple
import os
#%%

if __name__ == "__main__":
    
    ############# ATENÇÃO!!! EXEMPLO ADAPTADO PARA O CASO ESTÁTICO!!! #################
    ############### MATERIAL, AMORTECIMENTO E ESPESSURA GLOBAL (SI) ####################
    E = 210e9
    rho = 7850
    v = 0.3
    alpha_x = 0.11
    alpha_y = 0.70
    rho_air = 1.18
    v_air = 1.48e-5
    d_ = 0.024
    eta = 0.005
    U0 = 44.7
    # Proportinal damping
    alpha = 0
    beta = 1e-6
    # Espessura
    h_init = 0.00159
    #h_max = 0.005
    #h_min = 0.002
    #################### MALHA QUAD4 RETANGULO ##################################
    lx = 0.47 # Comprimento
    ly = 0.37 # Altura
    #vmax = 0.5*h_max*lx*ly   # Volume máximo = metade do volume máximo possível
    tamanho_elemento = 0.01
    coord, connect, nodes_faceX1, nodes_faceX2, nodes_faceY1, nodes_faceY2, nodes_middle = msh.malha2D_QUAD4(lx,ly,tamanho_elemento)
    nnode = len(coord)
    nel = len(connect)
    h = h_init*np.ones((nel, 1))
    # "faces" : linhas de contorno do retângulo. Saem com índices partindo do zero
    # "connect": sai com índices partindo do 1

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


    #%%
    # Frequências e constantes
    freq = np.logspace(1, 3, 100)
    Vr = modal_shape[z_dofs,:]  # Modos normais
    wn = 2 * np.pi * natural_frequencies[z_dofs]  # Frequências naturais (rad/s)
    glf = int(check_dof)  # Grau de liberdade da força
    w = 2 * np.pi * freq  # Frequências angulares

    # Inicializando as FRFs
    Rp_m = np.zeros(len(freq), dtype=complex)
    Rt_m = np.zeros((len(z_dofs), len(freq)), dtype=complex)
    eta = 0.005
    # Pré-calculando termos repetidos para otimização
    eta_wn2 = eta * wn**2  # Termo de amortecimento modal
    wn2 = wn**2  # Frequência natural ao quadrado

    # Vetorização dos cálculos das FRFs
    for glr in range(len(z_dofs)):
        # Pré-calculando para cada grau de liberdade de resposta
        for k in range(modes):
            # Cálculo vetorizado para todas as frequências de uma vez
            den = (1j*w)/ (wn2[k] - w**2 + 1j * eta_wn2[k])
                        
            # FRF pontual (mesmo ponto de força)
            Rp_m += (Vr[glf, k]**2) * den
            
            # FRF de transferência (resposta em outro ponto)
            Rt_m[glr, :] += Vr[glr, k] * Vr[glf, k] * den



    #%%
    alpha_x = 0.11
    alpha_y = 0.70
    rho_air = 1.18
    v_air = 1.48e-5
    d_ = 0.024
    eta = 0.005
    U0 = 44.7
    Re_d = 8 * U0*d_/v_air
    tau_w = 0.0225 * rho_air * U0**2/Re_d
    D = E * h**3/(12*(1-v**2))
    c0 = 343
    #%%
    import numpy as np
    from joblib import Parallel, delayed

    # Pré-computando constantes fora dos loops
    f = np.logspace(1, 3, 100)  # Frequências
    w_array = 2 * np.pi * f  # Frequências angulares
    Uc_array = U0 * (0.59 + 0.30 * np.exp(-0.89 * w_array * d_ / U0))  # Pré-computando Uc
    phi_pp_array = np.array((tau_w**2 * d_ / U0) * (5.1 / (1 + 0.44 * (w_array * d_ / U0)**(7/3))), ndmin=1)

    # Coordenadas x e y para diferença
    ksix = coord[:, None, 0] - coord[None, :, 0]  # Diferenças em x
    ksiy = coord[:, None, 1] - coord[None, :, 1]  # Diferenças em y

    Aij = (tamanho_elemento / 2) ** 2  # Pré-computando Aij
    K = stif_matrix[z_dofs, :]
    M = mass_matrix[z_dofs, :]

    # Pré-alocando memória para Gf e Gamma_w
    Gf = np.zeros((nnode, f.size), dtype=np.complex64)
    Gamma_w = np.zeros((nnode, nnode, f.size), dtype=np.complex64)
    H = np.ones((nnode, 1), dtype=np.complex64)

    # Função que será paralelizada
    def compute_for_frequency(kk, w, phi_pp, Uc, ksix, ksiy, Aij):
        ksix_Uc = np.abs(w * ksix / Uc)
        ksiy_Uc = np.abs(w * ksiy / Uc)

        # Calculando termos exponenciais e Gamma
        exp_x = np.exp(-alpha_x * ksix_Uc)  # Parte exponencial para ksix
        exp_y = np.exp(-alpha_y * ksiy_Uc)  # Parte exponencial para ksiy
        complex_exp = np.exp(1j * w * ksix / Uc)  # Parte exponencial complexa

        Gamma = (1 + alpha_x * ksix_Uc) * exp_x * complex_exp * exp_y
        Gamma_w[:, :, kk] = Gamma  # Armazenando em Gamma_w

        # Calculando o valor final para Gf
        Gf_kk = H.conj().T @ (Aij * phi_pp * Gamma * Aij) @ H
        return Gf_kk

    # Usando joblib para paralelizar o loop
    results = Parallel(n_jobs=-1)(delayed(compute_for_frequency)(
        kk, w, phi_pp, Uc, ksix, ksiy, Aij) for kk, (w, phi_pp, Uc) in enumerate(zip(w_array, phi_pp_array, Uc_array)))

    # Atualizando Gf com os resultados paralelos
    for kk, result in enumerate(results):
        Gf[:, kk] = result



    #%%
    # Precompute constants outside the loops
    f = np.logspace(1, 3, 100)
 
    Gf = np.zeros((nnode, f.size), dtype=np.complex64)

    #G = []
    #Gamma_w = np.zeros((f.size), dtype=np.complex64)
    Gamma_w = np.zeros((nnode,nnode, (f.size)), dtype=np.complex64)
    #Uc = 0.7*U0
    w_array = 2 * np.pi * f  # Array of angular frequencies
    Uc_array = U0 * (0.59 + 0.30 * np.exp(-0.89 * w_array * d_ / U0))  # Precompute Uc for all w
    phi_pp_array = np.array((tau_w**2 * d_ / U0) * (5.1 / (1 + 0.44 * (w_array * d_ / U0)**(7/3))), ndmin=1)
    # Coordinate difference arrays
    ksix = coord[:, None, 0] - coord[None, :, 0]  # Differences in x
    ksiy = coord[:, None, 1] - coord[None, :, 1]  # Differences in y

    Aij = (tamanho_elemento/2)**2  # Precompute Aij outside the loop

    # Loop over frequencies
    for kk, (w,  phi_pp, Uc) in enumerate(zip(w_array,  phi_pp_array, Uc_array)):
        # Precompute the normalized ksix and ksiy terms
        
        ksix_Uc = np.abs(w * ksix / Uc)
        ksiy_Uc = np.abs(w * ksiy / Uc)

        # Precompute the Gamma term
        exp_x = np.exp(-alpha_x * ksix_Uc)  # Exponential part for ksix
        exp_y = np.exp(-alpha_y * ksiy_Uc)  # Exponential part for ksiy
        complex_exp = np.exp(1j * w * ksix / Uc)  # Complex exponential part

        Gamma = (1 + alpha_x * ksix_Uc) * exp_x * complex_exp * exp_y
        
        Gamma_w[:,:,kk] = Gamma
        


    #%%
    PHI_p = np.zeros_like(freq)   
    for ii in range(100):
        PHI_p[ii] = np.conj(Rt_m[:,ii]).T@(phi_pp_array[ii]*Gamma_w[:,:,ii])@Rt_m[:,ii]
    
    plt.semilogy(freq, PHI_p)
    #plt.xlim(0,600)
    #ff = np.where(f==200)
    #plt.imshow(np.abs(G[0]), origin='lower', extent=(0,lx, 0, ly))

    #%%
    load_dofs_z = (np.array([2]) + 5*np.arange(nnode)).astype(int)
    F = np.zeros((5*nnode,f.size), dtype=complex)  
    F[load_dofs_z,:] = Gf
  
      
    


    # ################# ANÁLISE HARMÔNICA ##########################################
    PLOT_FRF = 1
    freq = f
    omega = 2*np.pi*freq[0]
    U = np.zeros((5*nnode,len(freq)),dtype=complex)
    
    U[free_dofs,:] = solv.harmonic_analysys(freq, stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], alpha, beta, F[free_dofs,:])
    #%%
    ################### PLOTS #######################################################
    # Plotar resposta de qualquer um dos graus de liberdade com carregamento
    dof_plot = check_dof
    fig, ax = plt.subplots()
    ax.semilogy(freq, abs(U[dof_plot,:].T), label='ABS U Tip')
    
    plt.show()

    #%%
    # Plotar campo de deslocamentos de um dado modo e de uma dada frequência específica
    pos = np.where(freq == 5000)[0]
    mode = 10

    Hv = np.ones((len(all_dofs),1))*2*np.pi*freq*1j
    V = Hv*U

    fig, ax = plt.subplots()
    ax.semilogy(freq, abs(U[check_node,:].T), label='ABS U Tip')
    plt.xlim(0,600)



    




    #%%
    vtk_write_displacement(U, freq, coord, connect, nnode, nel)
    vtk_write_velocity(V, freq, coord, connect, nnode, nel)
    vtk_write_modal(modal_shape, natural_frequencies, coord, connect, nnode, nel)
    
    
# %%
