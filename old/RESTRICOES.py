import numpy as np

def restricao(m,n,lx,ly,nel,h,vmax):
    #
    a_el = lx*ly/nel
    fval = np.zeros((m, 1))
    dfval = np.zeros((m,n))
    for el in range(n):
        fval[0,0] += a_el*h[el,0]
        dfval[0,el] = a_el

    return fval, dfval


