import numpy as np
from scipy.sparse import csr_matrix
import MESH as msh

def shapeQ4(ssx, ttx):
    """ Linear Shape Functions and Derivatives.
    """
    #shape functions 
    phi = np.zeros(4)
    #
    phi[0]=(1-ssx)*(1-ttx)/4
    phi[1]=(1+ssx)*(1-ttx)/4
    phi[2]=(1+ssx)*(1+ttx)/4
    phi[3]=(1-ssx)*(1+ttx)/4

    #derivatives
    dphi = np.zeros((2,4))
    #
    dphi[0,0]=(-1)*(1-ttx)/4
    dphi[0,1]=(+1)*(1-ttx)/4
    dphi[0,2]=(+1)*(1+ttx)/4
    dphi[0,3]=(-1)*(1+ttx)/4
    
    dphi[1,0]=(1-ssx)*(-1)/4
    dphi[1,1]=(1+ssx)*(-1)/4
    dphi[1,2]=(1+ssx)*(+1)/4
    dphi[1,3]=(1-ssx)*(+1)/4
             
    return phi, dphi

def matricesP4(ee, coord, connect, E, vv, rho, H):
    """ P4 stiffness and mass matrices.
        Simple: midplane, no integration over the thickness
    """
    ########### Pontos de Integração ############
    # Demais matrizes                 
    nint, con, wps = 4, 1/np.sqrt(3), 1
    pint = np.array([[-con, -con],
                      [ con, -con],
                      [ con,  con],
                      [-con,  con]])
    
    ############ Matrizes Constitutivas ############
    Dp = E/(1-(vv**2))*np.array([[ 1, vv, 0        ],
                                 [ vv,  1, 0       ],
                                 [ 0,   0, (1-vv)/2]])
    
    Dm = H*Dp  
    ##################### INTEGRAÇÃO ########################
    Ke, Me = 0, 0
    Bm = np.zeros((3,8))
    N = np.zeros((2,8))
    # integration
    for i in range(nint):
        ssx, ttx = pint[i, 0], pint[i, 1]
        phi, dphi = shapeQ4(ssx,ttx)
        ie = connect[ee,:]-1
        dxdy = dphi@coord[ie, 0:2] 
        # note: dxdr, dydr, dzdr, dxds, dyds, dzds, dxdt, dydt, dzdt 
        JAC = np.array([[dxdy[0,0], dxdy[0,1]],
                        [dxdy[1,0], dxdy[1,1]]], dtype=float) 
        detJAC = np.linalg.det(JAC)
      
        #Inverse Jacobian
        iJAC = np.linalg.inv(JAC) 
        
        dphi_t = iJAC @ dphi
        
        for iii in range(4):
            Bm[0,2*(iii)+0]=dphi_t[0,iii]
            Bm[0,2*(iii)+1]=0
             #
            Bm[1,2*(iii)+0]=0
            Bm[1,2*(iii)+1]=dphi_t[1,iii]
            #
            Bm[2,2*(iii)+0]=dphi_t[1,iii]
            Bm[2,2*(iii)+1]=dphi_t[0,iii]
            
        for iii in range(4):
            N[0,2*(iii)+0]=phi[iii]
            N[1,2*(iii)+1]=phi[iii]

        Ke += (Bm.T@Dm@Bm)*(detJAC*wps)
        Me += H*rho*N.T@N*(detJAC*wps)

    return Ke, Me

def generate_ind_rows_cols(connect):
    # processing the dofs indices (rows and columns) for assembly
    dofs, edofs = 2, 8
    ind_dofs =      (np.array([dofs*connect[:,0]-2, dofs*connect[:,0]-1,
                               dofs*connect[:,1]-2, dofs*connect[:,1]-1,
                               dofs*connect[:,2]-2, dofs*connect[:,2]-1,
                               dofs*connect[:,3]-2, dofs*connect[:,3]-1], dtype=int)).T
    vect_indices = ind_dofs.flatten()
    ind_rows = ((np.tile(vect_indices, (edofs,1))).T).flatten()
    ind_cols = (np.tile(ind_dofs, edofs)).flatten()
    return ind_rows, ind_cols

def stif_mass_matrices_0(coord, connect, nnode, nel, ind_rows, ind_cols, E, vv, rho, h, p, q, xval, THRSHLD):
    """ Calculates global matrices.
    """
    ngl = 2 * nnode
    data_k_ = np.zeros((nel, 8, 8), dtype=float)
    data_m_ = np.zeros((nel, 8, 8), dtype=float)
    data_k = np.zeros((nel, 64), dtype=float)
    data_m = np.zeros((nel, 64), dtype=float)
    
    for el in range(nel):
        x = xval[el,0]
        if THRSHLD != 0 and x <= 0.1:
            q = 9
        else:
            q = 1
        Ke, Me = matricesP4(el, coord, connect, E, vv, rho, h)
        data_k_[el,:,:] = Ke
        data_m_[el,:,:] = Me
        data_k[el,:] = (x**p)*Ke.flatten() 
        data_m[el,:] = (x**q)*Me.flatten() 
   
    data_k = data_k.flatten()
    data_m = data_m.flatten()
    stif_matrix = csr_matrix((data_k, (ind_rows, ind_cols)), shape=(ngl, ngl))
    mass_matrix = csr_matrix((data_m, (ind_rows, ind_cols)), shape=(ngl, ngl))

    return stif_matrix, mass_matrix, data_k_, data_m_

def stif_mass_matrices_OPT(data_k_, data_m_, nnode, nel, ind_rows, ind_cols, p, q, xval, THRSHLD):
    """ Calculates global matrices.
    """
    ngl = 2 * nnode
    data_k = np.zeros((nel, 64), dtype=float)
    data_m = np.zeros((nel, 64), dtype=float)
    
    for el in range(nel):
        x = xval[el,0]
        if THRSHLD != 0 and x <= 0.1:
            q = 9
        else:
            q = 1
        data_k[el,:] = (x**p)*data_k_[el,:,:].flatten() 
        data_m[el,:] = (x**q)*data_m_[el,:,:].flatten()
   
    data_k = data_k.flatten()
    data_m = data_m.flatten()
    stif_matrix = csr_matrix((data_k, (ind_rows, ind_cols)), shape=(ngl, ngl))
    mass_matrix = csr_matrix((data_m, (ind_rows, ind_cols)), shape=(ngl, ngl))

    return stif_matrix, mass_matrix