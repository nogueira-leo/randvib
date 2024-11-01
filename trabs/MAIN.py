# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import SHELL4 as fems4
import SOLVE as solv
import MESH as msh

from joblib import Parallel, delayed
import pandas as pd
import matplotlib
from numpy.linalg import norm
import os
matplotlib.use('qtagg')


#  %%

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
    
    
    # Propriedades
    E = 200e9
    rho = 7850
    v = 0.33
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
    c0 = 343
    
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
    
    ###################### ASSEMBLY #################################
    ind_rows, ind_cols = fems4.generate_ind_rows_cols(connect)
    stif_matrix, mass_matrix = fems4.stif_mass_matrices(coord, connect, nnode, nel, ind_rows, ind_cols, E, v, rho, h)
    
    ################### ANÁLISE MODAL ###############################
    modes = 5
    modal_shape = np.zeros((5*nnode,modes))
    natural_frequencies, modal_shape[free_dofs,:] = solv.modal_analysis(stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], modes, which='LM', sigma=0.01)
    print('Frequências Naturais:')
    print(natural_frequencies)
    # %%
    freq = np.linspace(50,2000,100)
    psd = pd.DataFrame(index=freq, dtype=float)
    csd = pd.DataFrame(index=freq, dtype=np.complex64)
    Vr = modal_shape[z_dofs,:]  # Modos normais
    wn = 2 * np.pi * natural_frequencies  # Frequências naturais (rad/s)
    #glf = check_node[0]  # Grau de liberdade da força
    w_array = 2 * np.pi * freq  # Frequências angulares
    k0_array = w_array/c0

    # Diferenças de coordenadas
    ksix = coord[:, None, 0] - coord[None, :, 0]  # Diferenças em x
    ksiy = coord[:, None, 1] - coord[None, :, 1]  # Diferenças em y
    ksin = norm(np.stack((ksix,ksiy)),2,0)
    np.fill_diagonal(ksin,1)
    # %%
    
    def compute_Gamma_DAF(kk,  k0,  ksin):
        Gamma = np.sin(k0 * ksin) / (k0 * ksin)
        return Gamma, kk

    # Preallocate arrays
    Gamma_DAF = np.zeros((nnode, nnode, len(w_array)), dtype=np.complex64)

    # Parallel computation of phi_pb using joblib
    results_DAF = Parallel(n_jobs=-1)(delayed(compute_Gamma_DAF)(
        kk, k0, ksin) for kk, k0 in enumerate(k0_array))

    # Combine the results back into phi_pb
    for Gamma, kk in tqdm(results_DAF):
        Gamma_DAF[:, :, kk] = Gamma

    #%%
    for n_eta, eta in enumerate(tqdm([0.5, 0.05, 0.005])):

        # Inicializando as FRFs    
        Hv = np.zeros((len(z_dofs), len(z_dofs), len(freq)), dtype=np.complex64)
        
        # Pré-calculando termos repetidos para otimização
        eta_wn2 = eta * wn**2  # Termo de amortecimento modal
        wn2 = wn**2  # Frequência natural ao quadrado

        # Function to compute the FRF for a specific (glf, glr) pair
        def compute_FRF(glr, glf, w_array, wn2, eta_wn2, Vr, modes):
            # Initialize the result for this pair
            Hv_glr_glf = np.zeros_like(w_array, dtype=np.complex64)
            
            # Calculate the denominator and FRF for each mode
            for k in range(modes):
                # Vectorized calculation of the denominator for all frequencies
                den = (1j * w_array) / (wn2[k] - w_array**2 + 1j * eta_wn2[k])
                
                # Compute the FRF and add the contribution for mode `k`
                Hv_glr_glf += Vr[glr, k] * Vr[glf, k] * den
            
            return Hv_glr_glf, glr, glf
        # Preallocate the array to store the final Hv
        Hv = np.zeros((nnode, nnode, len(w_array)), dtype=np.complex64)

        # Parallel computation of FRFs using joblib
        results = Parallel(n_jobs=-2)(delayed(compute_FRF)(
            glr, glf, w_array, wn2, eta_wn2, Vr, modes) for glr in np.arange(nnode) for glf in check_node)

        # Aggregate the results back into Hv
        for Hv_glr_glf, glr, glf in tqdm(results):
            Hv[glr, glf, :] = Hv_glr_glf
        
        for n_U0, U0 in enumerate(tqdm([44.7, 89.4, 178.8])):
            Re_d = 8 * U0*d_/v_air
            tau_w = (0.0225 * rho_air * U0**2)/(Re_d**0.25)
            # Pré-computando constantes fora dos loops
            w_array = 2 * np.pi * freq  # Frequências angulares
            Uc_array = U0 * (0.59 + 0.30 * np.exp(-0.89 * w_array * d_ / U0))  # Precomputando Uc para todas as frequências
            phi_pp = np.array((tau_w**2 * d_ / U0) * (5.1 / (1 + 0.44 * (w_array * d_ / U0)**(7/3))), ndmin=1)

            # Diferenças de coordenadas
            ksix = coord[:, None, 0] - coord[None, :, 0]  # Diferenças em x
            ksiy = coord[:, None, 1] - coord[None, :, 1]  # Diferenças em y
            Aij = (tamanho_elemento)**2  # Precomputando Aij fora do loop
            Aij = 0.47*0.37/nnode  # Precomputando Aij fora do loop
            # Função paralelizada que será aplicada em cada frequência
            def compute_Gamma_TBL(kk, w, Uc, ksix, ksiy, alpha_x, alpha_y, Aij):
                ksix_Uc = np.abs(w * ksix / Uc)
                ksiy_Uc = np.abs(w * ksiy / Uc)
                # Precomputando o termo Gamma
                exp_x = np.exp(-alpha_x * ksix_Uc)  # Parte exponencial para ksix
                exp_y = np.exp(-alpha_y * ksiy_Uc)  # Parte exponencial para ksiy
                complex_exp = np.exp(1j * w * ksix / Uc)  # Parte exponencial np.complex64a

                Gamma = (1 + alpha_x * ksix_Uc) * exp_x * complex_exp * exp_y
                return Gamma, kk
            

            # Usando joblib para paralelizar o loop sobre frequências
            results_TBL = Parallel(n_jobs=-2)(delayed(compute_Gamma_TBL)(
                kk, w, Uc, ksix, ksiy, alpha_x, alpha_y, Aij) for kk, (w,  Uc) in enumerate(zip(w_array, Uc_array)))

            # Atualizando Gamma_w com os resultados paralelizados
            Gamma_TBL = np.zeros((nnode,nnode,len(freq)), dtype=np.complex64)   
            for Gamma, kk in tqdm(results_TBL):
                Gamma_TBL[:,:,kk] = Gamma

            

             
            Gvv_TBL = np.zeros_like(Gamma_DAF, dtype=np.complex64)
            Gvv_DAF = np.zeros_like(Gamma_DAF, dtype=np.complex64)
            Gpp_TBL = np.zeros_like(Gamma_DAF, dtype=np.complex64)
            Gpp_DAF = np.zeros_like(Gamma_DAF, dtype=np.complex64)

            
            for ii in tqdm(range(len(freq))):
                Gvv_TBL[:,:,ii] = np.conj(Hv[:,:,ii].T)@(phi_pp[ii]*Gamma_TBL[:,:,ii])@Hv[:,:,ii]*Aij
                Gvv_DAF[:,:,ii] = np.conj(Hv[:,:,ii].T)@(phi_pp[ii]*Gamma_DAF[:,:,ii])@Hv[:,:,ii]*Aij
                Gpp_TBL[:,:,ii] = phi_pp[ii]*Gamma_TBL[:,:,ii]
                Gpp_DAF[:,:,ii] = phi_pp[ii]*Gamma_DAF[:,:,ii]
                
            

            
            psd[f'psd_{n_eta}_{n_U0}'] = phi_pp
            csd[f'TBL_{n_eta}_{n_U0}'] = Gvv_TBL[check_node[0], check_node[0],:]
            csd[f'DAF_{n_eta}_{n_U0}'] = Gvv_DAF[check_node[0], check_node[0],:]

    # %% 
    eta = [0.5, 0.05, 0.005]
    U0  = [44.7, 89.4, 178.8]
    #eta = [0.05]
    #U0 = [44.7]
    
    # %%
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.title(rf"Densidade de Auto Espectro - $ \eta={eta[1]}$")
    plt.plot(10*np.log10(np.abs(psd[f'psd_{1}_{0}']/2e-5**2)),'k')
    plt.plot(10*np.log10(np.abs(psd[f'psd_{1}_{1}']/2e-5**2)),'--k')
    plt.plot(10*np.log10(np.abs(psd[f'psd_{1}_{2}']/2e-5**2)),':k')
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("PSD $(dB - ref\ 20 \mu Pa)$")
    plt.legend(U0)
    #plt.ylim(84.55,84.60)
    plt.grid(True, which='both')
    plt.show()



    # %%

    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.title(rf"Densidade Espectral Crusada - Nó ${check_node[0]}$ - $\eta={eta[1]}$")
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{0}']/1e-9**2)), 'r')
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{1}']/1e-9**2)), '--r')
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{2}']/1e-9**2)), ':r')
    plt.plot(10*np.log10(np.abs(csd[f'DAF_{1}_{0}']/1e-9**2)), 'b')
    plt.plot(10*np.log10(np.abs(csd[f'DAF_{1}_{1}']/1e-9**2)), '--b')
    plt.plot(10*np.log10(np.abs(csd[f'DAF_{1}_{2}']/1e-9**2)), ':b')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("CSD (dB ref $1nm$) ")
    plt.legend([f"TBL - $U_0 = {U0[0]} m/s$",
                f"TBL - $U_0 = {U0[1]} m/s$",
                f"TBL - $U_0 = {U0[2]} m/s$",
                f"DAF - $U_0 = {U0[0]} m/s$",
                f"DAF - $U_0 = {U0[1]} m/s$",
                f"DAF - $U_0 = {U0[2]} m/s$"])
    plt.grid(True, which='both')
    plt.show()
    # %% 
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.title(rf"Densidade Espectral Crusada - Nó ${check_node[0]}$ - $U_0={U0[1]} m/s$")
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{0}_{1}']/1e-9**2)), 'r')
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{1}']/1e-9**2)), '--r')
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{2}_{1}']/1e-9**2)), ':r')
    plt.plot(10*np.log10(np.abs(csd[f'DAF_{0}_{1}']/1e-9**2)), 'b')
    plt.plot(10*np.log10(np.abs(csd[f'DAF_{1}_{1}']/1e-9**2)), '--b')
    plt.plot(10*np.log10(np.abs(csd[f'DAF_{2}_{1}']/1e-9**2)), ':b')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("CSD (dB ref $1nm$) ")
    plt.legend([f"TBL - $\eta = {eta[0]}$",
                f"TBL - $\eta = {eta[1]}$",
                f"TBL - $\eta = {eta[2]}$",
                f"DAF - $\eta = {eta[0]}$",
                f"DAF - $\eta = {eta[1]}$",
                f"DAF - $\eta = {eta[2]}$"])
    plt.grid(True, which='both')
    plt.show()

    # %%
    


    # %%

    Gvvw_TBL = Gamma_TBL[check_node[0],:,:].T
    Gvvw_DAF = Gamma_DAF[check_node[0],:,:].T
    Gppw_TBL = Gpp_TBL[check_node[0],:,:].T
    Gppw_DAF = Gpp_DAF[check_node[0],:,:].T
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.imshow(np.abs(Gvvw_TBL), origin='lower', extent=(1, nnode+1, freq[0], freq[-1]), aspect='auto')
    plt.title(rf"Coerência Espacial do Nó ${check_node[0]}$ - TBL")
    plt.xlabel('Id do Nó')
    plt.ylabel('Frequência (Hz)')
    plt.colorbar()
    plt.show()
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.imshow(np.abs(Gvvw_DAF), origin='lower', extent=(1, nnode+1, freq[0], freq[-1]), aspect='auto')
    plt.title(rf"Coerência Espacial do Nó ${check_node[0]}$ - DAF")
    plt.xlabel('Id do Nó')
    plt.ylabel('Frequência (Hz)')
    plt.colorbar()
    plt.show()
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.imshow(np.abs(Gppw_TBL), origin='lower', extent=(1, nnode+1, freq[0], freq[-1]), aspect='auto')
    plt.title(rf"Densidade Espectral Cruzada do Nó ${check_node[0]}$ - TBL")
    plt.xlabel('Id do Nó')
    plt.ylabel('Frequência (Hz)')
    plt.colorbar()
    plt.show()
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.imshow(np.abs(Gppw_DAF), origin='lower', extent=(1, nnode+1, freq[0], freq[-1]), aspect='auto')
    plt.title(rf"Densidade Espectral Cruzada do Nó ${check_node[0]}$ - DAF")
    plt.xlabel('Id do Nó')
    plt.ylabel('Frequência (Hz)')
    plt.colorbar()
    plt.show()
    # %%
    plt.figure(figsize=(16,9), dpi=200, layout='tight')
    plt.title(rf"Densidade Espectral Crusada - Nó ${check_node[0]}$ - $\eta={eta[1]}$")
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{0}']/1e-9**2)), 'r')
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{1}']/1e-9**2)), '--r')
    plt.plot(10*np.log10(np.abs(csd[f'TBL_{1}_{2}']/1e-9**2)), ':r')
    plt.plot(f_0, 10*np.log10(np.abs(Gvv_0/1e-9**2)), 'g')
    plt.plot(f_1, 10*np.log10(np.abs(Gvv_1/1e-9**2)), '--g')
    plt.plot(f_2, 10*np.log10(np.abs(Gvv_2/1e-9**2)), ':g')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('CSD (dB ref $1nm$) ')
    plt.legend([rf"Nogueira - $\eta = {eta[1]}, U_0 = {U0[0]}$",
                rf"Nogueira - $\eta = {eta[1]}, U_0 = {U0[1]}$",
                rf"Nogueira - $\eta = {eta[1]}, U_0 = {U0[2]}$",
                rf"Hambric - $\eta = {eta[1]}, U_0 = {U0[0]}$",
                rf"Hambric - $\eta = {eta[1]}, U_0 = {U0[1]}$",
                rf"Hambric - $\eta = {eta[1]}, U_0 = {U0[2]}$"])
    plt.grid(True)
    plt.show()






    # %%
    #plt.figure(figsize=(16,9), dpi=200, layout='tight')
    #plt.title(rf"FRFs - do - Nó ${check_node[0]}$")
    #plt.plot(freq,10*np.log10(np.abs(Hv.T)),'lightgray')
    #plt.plot(freq,10*np.log10(np.abs(Hv[check_node[0],:].T)),'k')
#
    #plt.xlabel('Frequência (Hz)')
    #plt.ylabel('Mobilidade (dB)')
    #plt.show()
# %%
