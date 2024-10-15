import numpy as np
from scipy.sparse.linalg import eigs, spsolve
from pypardiso.pardiso_wrapper import PyPardisoSolver
from scipy.sparse import triu, csr_matrix
import time

" Olavo M Silva 2024 MOPT"

def modal_analysis(K=[], M=[], modes=20, which='LM', sigma=0.01, normalize=True):
    """
    """
    KT = K
    MT = M
       
    eigen_values, eigen_vectors = eigs(KT, M=MT, k=modes, which=which, sigma=sigma)

    positive_real = np.absolute(np.real(eigen_values))
    natural_frequencies = np.sqrt(positive_real)/(2*np.pi)
    modal_shape = np.real(eigen_vectors)

    index_order = np.argsort(natural_frequencies)
    natural_frequencies = natural_frequencies[index_order]
    modal_shape = modal_shape[:, index_order]
    
    if normalize:
        modal_shape /= np.max(np.abs(modal_shape), axis=0)

    return natural_frequencies, modal_shape

def harmonic_analysys(freq, K, M, alpha, beta, F):
    ps = PyPardisoSolver(mtype=6)
    print('Rodando Análise Harmônica...')
    U = np.empty([len(F),len(freq)]).astype(complex)
    counter_1 = 0
    for a in freq:
        omega = 2 * (np.pi) * a
        C = alpha*M + beta*K
        A = K - (omega**2)*M +1j*omega*C
        A = triu(A, format="csr")
        #t = time.time()
        #print('Solving freq: ',a)
        U_aux = ps.solve(A, F)
        ps.free_memory(everything=True)
        #elapsed = time.time() - t
        #print('=====> Elapsed time: ',elapsed)
        U[:,counter_1] = U_aux[:,0]
        counter_1 +=  1
    print("Análise Harmônica Finalizada com Sucesso!")
    return U