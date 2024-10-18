import numpy as np




def frequency_range(f_min, f_max, num_points):
    # Generate a logarithmic range of frequencies
    return np.logspace(np.log10(f_min), np.log10(f_max), num_points)

def corcos_model(f, U0, dx, dy):
    # Calculate the cross-spectral matrix according to the Corcos model
    #"$S_{pp}(f,\Delta x,\Delta y)=S_{pp}(f)e^{-\alpha_x\frac{f\Delta x}{U}}e^{-\alpha_y\frac{f\Delta y}{U}}$"
    #alphax = 0.1 * (1 + 0.02 * f)  # Example modification
    #alphay = 0.1 * (1 + 0.01 * f)  # Example modification
    
    CSD = np.exp(-alpha_x * (f * dx) / U0) * np.exp(-alpha_y * (f * dy) / U0)
    return CSD


def Gamma(ksix,ksiy, w):
    A = (1+ alpha_x * np.abs(w*ksix/Uc))*np.exp(-alpha_x * np.abs(w*ksix/Uc))*np.exp(1j*w*ksix/Uc)
    B = np.exp(-alpha_y * np.abs(w * ksiy/Uc))