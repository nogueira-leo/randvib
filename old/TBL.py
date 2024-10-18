import numpy as np


def corcos_model(f, dx, dy):
    # Calculate the cross-spectral matrix according to the Corcos model
    uc = theta[0]
    alphax = theta[1]
    alphay = theta[2]
    f = theta[3]
    S = np.exp(-2 * np.pi * f / uc * (alphax * np.abs(dx) + alphay * np.abs(dy) - 1j * dx))
    S = (S + S.T) / 2
    return S