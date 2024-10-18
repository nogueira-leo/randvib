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

def matricesQ4(ee, coord, connect, E, vv, rho, H):
    """ Q4 stiffness and mass matrices.
        Simple: midplane, no integration over the thickness
    """
    cor_cis = 5/6

    ########### Pontos de Integração ############
    # Cisalhamento
    nint_s = 1
    wps_s = 2
    pint_s = np.array([[0, 0]])
    # Demais matrizes                 
    nint, con, wps = 4, 1/np.sqrt(3), 1
    pint = np.array([[-con, -con],
                      [ con, -con],
                      [ con,  con],
                      [-con,  con]])
    
    ############### Matrizes Constitutivas ##################
    G = E/(2*(1+vv))

    Dp = E/(1-(vv**2))*np.array([[ 1,  vv,  0       ],
                                 [ vv,  1,  0       ],
                                 [ 0,   0,  (1-vv)/2]])
    
    Dm = H*Dp  
    dDm = Dp  ##### ---> Para derivada

    Ds = G*np.array([[ 1, 0],
                     [ 0, 1]])
    
    Db = ((H**3)/12)*Dp
    dDb = (3*(H**2)/12)*Dp ##### ---> Para derivada


    Ds = cor_cis*H*Ds
    dDs = cor_cis*Ds ##### ---> Para derivada

    II =      rho*np.array([[ H, 0, 0,         0,         0],
                            [ 0, H, 0,         0,         0],
                            [ 0, 0, H,         0,         0],
                            [ 0, 0, 0, (H**3)/12,         0],
                            [ 0, 0, 0,         0, (H**3)/12]])
    
    dII =     rho*np.array([[ 1, 0, 0,           0,          0],
                            [ 0, 1, 0,           0,          0],
                            [ 0, 0, 1,           0,          0],
                            [ 0, 0, 0, 3*(H**2)/12,          0],
                            [ 0, 0, 0,          0, 3*(H**2)/12]])
    ##################### INTEGRAÇÃO ########################
    Ke, Me = 0, 0
    dKe, dMe = 0, 0
    AUJJ = np.zeros((2,2))
    Bb = np.zeros((3,20))
    Bm = np.zeros((3,20))
    Bs = np.zeros((2,20))
    N = np.zeros((5,20))
    # integration
    for i in range(nint):
        ssx, ttx = pint[i, 0], pint[i, 1]
        phi, dphi = shapeQ4(ssx,ttx)
        ie = connect[ee,:]-1
        dxdy = dphi@coord[ie, :] 
        # note: dxdr, dydr, dzdr, dxds, dyds, dzds, dxdt, dydt, dzdt 
        JAC = np.array([[dxdy[0,0], dxdy[0,1]],
                        [dxdy[1,0], dxdy[1,1]]], dtype=float) 
        detJAC = np.linalg.det(JAC)
      
        #Inverse Jacobian
        iJAC = np.linalg.inv(JAC) 
        
        dphi_t = iJAC @ dphi
        
        for iii in range(4):
            Bb[0,5*(iii)+0]=0
            Bb[0,5*(iii)+1]=0
            Bb[0,5*(iii)+2]=0
            Bb[0,5*(iii)+3]=0
            Bb[0,5*(iii)+4]=dphi_t[0,iii]
            #
            Bb[1,5*(iii)+0]=0
            Bb[1,5*(iii)+1]=0
            Bb[1,5*(iii)+2]=0
            Bb[1,5*(iii)+3]=-dphi_t[1,iii]
            Bb[1,5*(iii)+4]=0
            #
            Bb[2,5*(iii)+0]=0
            Bb[2,5*(iii)+1]=0
            Bb[2,5*(iii)+2]=0
            Bb[2,5*(iii)+3]=-dphi_t[0,iii]
            Bb[2,5*(iii)+4]=dphi_t[1,iii]
            #########################################
            Bm[0,5*(iii)+0]=dphi_t[0,iii]
            Bm[0,5*(iii)+1]=0
            Bm[0,5*(iii)+2]=0
            Bm[0,5*(iii)+3]=0
            Bm[0,5*(iii)+4]=0
            #
            Bm[1,5*(iii)+0]=0
            Bm[1,5*(iii)+1]=dphi_t[1,iii]
            Bm[1,5*(iii)+2]=0
            Bm[1,5*(iii)+3]=0
            Bm[1,5*(iii)+4]=0
            #
            Bm[2,5*(iii)+0]=dphi_t[1,iii]
            Bm[2,5*(iii)+1]=dphi_t[0,iii]
            Bm[2,5*(iii)+2]=0
            Bm[2,5*(iii)+3]=0
            Bm[2,5*(iii)+4]=0
            
        for iii in range(4):
            N[0,5*(iii)+0]=phi[iii]
            N[1,5*(iii)+1]=phi[iii]
            N[2,5*(iii)+2]=phi[iii]
            N[3,5*(iii)+3]=phi[iii]
            N[4,5*(iii)+4]=phi[iii]

        Ke += (Bm.T@Dm@Bm + Bb.T@Db@Bb)*(detJAC*wps)
        dKe += (Bm.T@dDm@Bm + Bb.T@dDb@Bb)*(detJAC*wps) # Para derivada
        Me += N.T@II@N*(detJAC*wps)
        dMe += N.T@dII@N*(detJAC*wps)

    
    for i in range(nint_s):
        ssx, ttx = pint_s[i, 0], pint_s[i, 1]
        phi, dphi = shapeQ4(ssx,ttx)
        ie = connect[ee,:]-1
        dxdy = dphi@coord[ie, :] 
        # note: dxdr, dydr, dzdr, dxds, dyds, dzds, dxdt, dydt, dzdt 
        JAC = np.array([[dxdy[0,0], dxdy[0,1]],
                        [dxdy[1,0], dxdy[1,1]]], dtype=float) 
        detJAC = np.linalg.det(JAC)
      
        #Inverse Jacobian
        iJAC = np.linalg.inv(JAC) 
        
        dphi_t = iJAC @ dphi
        
        for iii in range(4):
            Bs[0,5*(iii)+0]=0
            Bs[0,5*(iii)+1]=0
            Bs[0,5*(iii)+2]=dphi_t[1,iii]
            Bs[0,5*(iii)+3]=-phi[iii]
            Bs[0,5*(iii)+4]=0
            #
            Bs[1,5*(iii)+0]=0
            Bs[1,5*(iii)+1]=0
            Bs[1,5*(iii)+2]=dphi_t[0,iii]
            Bs[1,5*(iii)+3]=0
            Bs[1,5*(iii)+4]=phi[iii]

        Ke += (Bs.T@Ds@Bs)*(detJAC*wps_s)
        dKe += (Bs.T@dDs@Bs)*(detJAC*wps_s)

    return Ke, Me, dKe, dMe

def generate_ind_rows_cols(connect):
    # processing the dofs indices (rows and columns) for assembly
    dofs, edofs = 5, 20
    ind_dofs =      (np.array([dofs*connect[:,0]-5, dofs*connect[:,0]-4, dofs*connect[:,0]-3, dofs*connect[:,0]-2, dofs*connect[:,0]-1,
                               dofs*connect[:,1]-5, dofs*connect[:,1]-4, dofs*connect[:,1]-3, dofs*connect[:,1]-2, dofs*connect[:,1]-1,
                               dofs*connect[:,2]-5, dofs*connect[:,2]-4, dofs*connect[:,2]-3, dofs*connect[:,2]-2, dofs*connect[:,2]-1,
                               dofs*connect[:,3]-5, dofs*connect[:,3]-4, dofs*connect[:,3]-3, dofs*connect[:,3]-2, dofs*connect[:,3]-1], dtype=int)).T
    vect_indices = ind_dofs.flatten()
    ind_rows = ((np.tile(vect_indices, (edofs,1))).T).flatten()
    ind_cols = (np.tile(ind_dofs, edofs)).flatten()
    return ind_rows, ind_cols

def stif_mass_matrices(coord, connect, nnode, nel, ind_rows, ind_cols, E, vv, rho, h):
    """ Calculates global matrices.
    """
    ngl = 5 * nnode
    data_k = np.zeros((nel, 400), dtype=float)
    data_m = np.zeros((nel, 400), dtype=float)
    
    for el in range(nel):
        H = h[el,0]
        Ke, Me, _, _ = matricesQ4(el, coord, connect, E, vv, rho, H)
        data_k[el,:] = Ke.flatten() 
        data_m[el,:] = Me.flatten() 
   
    data_k = data_k.flatten()
    data_m = data_m.flatten()
    stif_matrix = csr_matrix((data_k, (ind_rows, ind_cols)), shape=(ngl, ngl))
    mass_matrix = csr_matrix((data_m, (ind_rows, ind_cols)), shape=(ngl, ngl))

    return stif_matrix, mass_matrix