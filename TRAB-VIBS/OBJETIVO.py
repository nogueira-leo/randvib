import numpy as np
import SHELL4 as fems4
#
def objetivo(freq, alpha, beta, F, U, LAMBDA, mass_matrix, coord, connect, nel, E, vv, rho, h):

    # Função objetivo: potencia de entrada
    omega = 2*np.pi*freq[0]
    # Função objetivo: potência de entrada ativa
    #f0val = 0.5 * omega * np.real(1j*np.conj(F).T @ U)

    # Função objetivo: energia cinética médoa
    f0val = 0.25*(omega**2)*np.conj(U).T @ mass_matrix @ U
    f0val = f0val.real # tem 0j, depois preciso do valor puramente real

    # Derivada da função objetivo
    df0val = np.zeros((nel,1))
    for el in range(nel):
        H = h[el,0]
        dofs = 5
        ind_dofs =      (np.array([dofs*connect[el,0]-5, dofs*connect[el,0]-4, dofs*connect[el,0]-3, dofs*connect[el,0]-2, dofs*connect[el,0]-1,
                                   dofs*connect[el,1]-5, dofs*connect[el,1]-4, dofs*connect[el,1]-3, dofs*connect[el,1]-2, dofs*connect[el,1]-1,
                                   dofs*connect[el,2]-5, dofs*connect[el,2]-4, dofs*connect[el,2]-3, dofs*connect[el,2]-2, dofs*connect[el,2]-1,
                                   dofs*connect[el,3]-5, dofs*connect[el,3]-4, dofs*connect[el,3]-3, dofs*connect[el,3]-2, dofs*connect[el,3]-1], dtype=int)).T
        Ue = U[ind_dofs]
        lambda_e = LAMBDA[ind_dofs]

        _, _, dKe, dMe = fems4.matricesQ4(el, coord, connect, E, vv, rho, H)

        dCe = alpha*dMe + beta*dKe

        dKd = dKe - (omega**2)*dMe + 1j*omega*dCe

        #df0val[el,0] = -0.5 * omega * np.real(1j*(Ue.T @ dKd @ Ue))  #potência de entrada ativa

        temp = 0.25*(omega**2)*np.conj(Ue).T @ dMe @ Ue + np.real(lambda_e.T @ dKd @ Ue)

        df0val[el,0] = temp[0].real  #energia cinética  #tem 0j, mas preciso depois do numero puramente real
    
    return f0val[0,0],  df0val

def objetivo_freq(freq, F, mass_matrix, U_init, U_opt):

    f_init = np.zeros((len(freq),1),dtype=complex)
    f_opt = np.zeros((len(freq),1),dtype=complex)

    for i in range(len(freq)):
        omega = 2*np.pi*freq[i]
        f_init[i] = 0.25*(omega**2)*np.conj(U_init[:,i]).T @ mass_matrix @ U_init[:,i]
        f_opt[i] = 0.25*(omega**2)*np.conj(U_opt[:,i]).T @ mass_matrix @ U_opt[:,i]
       
    return f_init.real, f_opt.real
