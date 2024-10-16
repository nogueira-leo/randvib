# %% imports
import numpy as np
import scipy
from scipy.fft import fftn, fft
import matplotlib.pyplot as plt
import SHELL4 as fems4
import SOLVE as solv
import MESH as msh
from VTK_FUNCS import vtk_write_displacement, vtk_write_modal, vtk_write_velocity
from joblib import Parallel, delayed
import pandas as pd

#%%
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
    ############# ATENÇÃO!!! EXEMPLO ADAPTADO PARA O CASO ESTÁTICO!!! #################
    ############### MATERIAL, AMORTECIMENTO E ESPESSURA GLOBAL (SI) ####################
    E = 200e9
    rho = 8000
    v = 0.3
    alpha_x = 0.11
    alpha_y = 0.70
    rho_air = 1.201
    v_air = 1.33e-5
    d_ = 0.0024 
    # Proportinal damping
    alpha = 0
    beta = 1e-6
    # Espessura
    h_init = 0.0159
    eta = 0.05
    U0 = 89.4
    Re_d = 8 * U0*d_/v_air
    tau_w = 0.0225 * rho_air * U0**2/Re_d**0.25
    
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
    modes = 10
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
        Gxx = (phi_pp * Gamma * Aij**2)
        
        
        return Gxx, kk
    
    # Pré-computando constantes fora dos loops
    w_array = 2 * np.pi * freq  # Frequências angulares
    Uc_array = U0 * (0.59 + 0.30 * np.exp(-0.89 * w_array * d_ / U0))  # Precomputando Uc para todas as frequências
    phi_pp_array = np.array((tau_w**2 * d_ / U0) * (5.1 / (1 + 0.44 * (w_array * d_ / U0)**(7/3))), ndmin=1)

    # Diferenças de coordenadas
    ksix = coord[:, None, 0] - coord[None, :, 0]  # Diferenças em x
    ksiy = coord[:, None, 1] - coord[None, :, 1]  # Diferenças em y
    Aij = (tamanho_elemento/4)**2  # Precomputando Aij fora do loop

    # Usando joblib para paralelizar o loop sobre frequências
    results = Parallel(n_jobs=-1)(delayed(compute_force)(
        kk, w, phi_pp, Uc, ksix, ksiy, alpha_x, alpha_y, Aij) for kk, (w, phi_pp, Uc) in enumerate(zip(w_array, phi_pp_array, Uc_array)))


    	
    #%% Atualizando Gamma_w com os resultados paralelizados
    Gxx_w = np.zeros((nnode,nnode,len(freq)), dtype=complex)   
    for Gxx, kk in results:
        Gxx_w[:,:,kk] = Gxx
        
    #%%
    Gvv = np.zeros(len(freq), dtype=complex)
    Gv = np.zeros(len(freq), dtype=complex)
    H1 = np.ones_like(Hv)   
    
    
    for ii in range(len(freq)):
        Gvv[ii] = np.conj(Hv[:,ii].T)@(Gxx_w[:,:,ii])@Hv[:,ii]
        Gv[ii] = np.conj(H1[:,ii].T)@(Gxx_w[:,:,ii])@H1[:,ii]

       
        
        
        
    #%%
    
    plt.semilogy(freq,(np.abs(Gvv)))
    #plt.semilogy(freq,(np.abs(Gv*np.sum(np.abs(Hv), axis=0))))
    #plt.semilogy(f_0, (np.abs(Gvv_0)))
    plt.semilogy(f_1, (np.abs(Gvv_1)))
    #plt.semilogy(f_2, (np.abs(Gvv_2)))
    
    #plt.plot(freq, np.log10(abs((Guu))))
    #plt.plot(freq, np.log10(abs((Gxx))))
    #plt.semilogy(freq, abs(Guu))
    plt.grid(True, which='major')
    #plt.ylim(-19, -9)
    #plt.xlim(0,600)
    




