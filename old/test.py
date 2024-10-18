# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import j1
#%%################# MALHA QUAD4 RETANGULO ##################################
lx = 0.47 # Comprimento
ly = 0.37 # Altura
h = 0.00159
E = 210e9
rho = 7850
nu = 0.3
alpha_x = 0.11
alpha_y = 0.70
rho_air = 1.18
v_air = 1.48e-5
d_ = 0.024
eta = 0.005
U0 = 44.7
Re_d = 8 * U0*d_/v_air
tau_w = 0.0225 * rho_air * U0**2/Re_d
D = E * h**3/(12*(1-nu**2))
c0 = 343
#%%

f = np.linspace(10,2000,1000)
f = 2000
w = 2*np.pi*f
Uc = U0 * (0.59+0.30*np.exp(-0.89*w*d_/U0))
phi_pp = (tau_w**2 * d_/U0) * (5.1/(1+0.44*(w*d_/U0)**(7/3)))
ka = w/c0
kc = w/Uc
kb = w**0.5*(rho*h/D)**0.25
k = np.logspace(-3,3,10000)
m = 3
a = lx
km = m*np.pi/a
Smka2_C = 4/(1-(km*a)**-1) * ((km*a)**2/((km*a)**2 + (k*a)**2))**2 * ((np.sin((km-k)*a/2))/((km-k)*a) + (np.sin((km+k)*a/2))/((km+k)*a))**2
Smka2_S = 2*(km*a)**2*(1-(-1)**m*np.cos(k*a))/((k*a)**2 - (km*a)**2)**2
Smka2_F = 2*(k*a)**2 * (1-(-1)**m*np.cos(k*a))/((k*a)**2 - (km*a)**2)**2
Gk = 1/k*2*alpha_x**3/(np.pi**2*alpha_y**3)*phi_pp
Hup = -1j*w/(D*(k**4-kb**4*(1+1j*eta)))
plt.semilogx(k*U0/w, 10*np.log10(Smka2_S))
plt.semilogx(k*U0/w, 10*np.log10(Smka2_C))
plt.semilogx(k*U0/w, 10*np.log10(Smka2_F))
plt.xlim(0.001,2)
#plt.semilogx(k*Uc/w, 10*np.log10(Hup))
plt.grid(True, "both")
plt.legend(['Suported', 'Clamped', 'Free'])
plt.ylim([-120,0])








# %%
