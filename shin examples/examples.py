# %%
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import pi
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import chirp, butter, filtfilt, correlate, correlation_lags, lfilter, convolve, csd
from scipy.io import loadmat

# %% 7.1
N=1000
f_heads = np.zeros(N)
f_tails = np.zeros(N)
for n in np.arange(1, N):
    x = np.random.rand(n)
    x = np.round(x)
    heads_id = np.where(x==1)[0]
    tails_id = np.where(x==0)[0]
    
    heads = heads_id.size
    tails = tails_id.size
    f_heads[n] = heads/n
    f_tails[n] = tails/n

plt.plot(f_heads)
plt.plot(f_tails)

# %% 7.2

x = np.random.uniform(size=(10, 5000))

nbin = 20
N = x.size
# %%
for ii in range(1,10):
    S = np.sum(x[:ii,:], axis=0)
    plt.figure()
    plt.hist(S, bins=nbin)
# %% 7.3
def correlation(x, y):
    return np.sum((x-np.mean(x))*(y-np.mean(y))/(np.std(x)*np.std(y)))/x.size

# %%
n = 1000
a, b = 2, 3
X = np.random.randn(n)
Y1 = a*X+b
Y2 = np.random.randn(n)
Y3 = X+Y2
rho1 = correlation(X,Y1)
rho2 = correlation(X,Y3)
rho3 = correlation(X,Y2)
plt.figure()
plt.plot(X,Y1, 'o')
plt.title(rf'$\rho = {rho1}$')
plt.xlabel('X')
plt.ylabel('Y1')

plt.figure()
plt.plot(X,Y2, 'o')
plt.title(rf'$\rho = {rho2}$')
plt.xlabel('X')
plt.ylabel('Y2')

plt.figure()
plt.plot(X,Y3, 'o')
plt.title(rf'$\rho = {rho3}$')
plt.xlabel('X')
plt.ylabel('Y3')
# %% 7.4

def kurtosis(x):
    return np.sum((x-x.mean())**4)/(x.std()**4)/x.size-3
    


X = np.random.randn(10000)
k = kurtosis(X)
plt.figure()
plt.plot(X)
plt.figure()
plt.hist(X,50)
# %% 8.1
n = 10000
A = 2
w = 1
theta = np.random.uniform(size=n)*2*np.pi
x1 = A*np.sin(theta)
t = np.linspace(0,2*np.pi,n)
x2 = A*np.sin(w*t)
plt.figure()
plt.hist(x1,20)
plt.figure()
plt.hist(x2,20)
# %% 8.2
def xcorr(x, y, fs):
    """Cross-correlation of two signals."""
    #corr = np.correlate(x, y, mode='same')  # scale = 'none'
    corr = correlate(x,y, mode='full') # scale = 'none
    #lags = np.arange(-(x.size - 1), x.size)
    lags = correlation_lags(x.size, y.size)
    corr /= (x.size - abs(lags))
    return lags/fs, corr 
    
#%%

A = 2
w = 2*np.pi*1
fs = 1000
t = 0
theta = np.random.uniform(size=(5000,1))*2*np.pi
x1 = A*np.sin(w*t+theta)
maxlag = 5
tau = np.linspace(-maxlag, maxlag, fs)
Rxx1 = np.zeros_like(tau)
for ii, tt in enumerate(tau):
    tmp = A*np.sin(w*(t+tt)+theta)
    Rxx1[ii] = np.mean(x1*tmp)
Rxx = A**2/2*np.cos(w*tau)

plt.figure()
plt.plot(tau,Rxx1)
plt.plot(tau, Rxx, '--')


t = np.linspace(0,1 , fs, endpoint=False)
tau2 = np.linspace(-20,20 , 2*fs-1, endpoint=False)
x2 = A*np.sin(w*t)
tau2, RxX2 = xcorr(x2,x2,fs)


plt.figure()
#plt.plot(tau2, Rxx2)
plt.plot(tau2, RxX2)
# %% 8.3
a=2
b=1
fs=200
delta1=1
delta2=2.5
t=np.linspace(0,0.5,fs, endpoint=False)
s=np.sin(2*pi*10*t)
N=4*fs
x = np.zeros(N)
x[fs:2*fs] = a*s
x[int(2.5*fs):int(3.5*fs)] = b*s

t = np.linspace(0,4,x.size)
Rxx = correlate(x,x, mode='full')
tau = correlation_lags(x.size,x.size, mode='full')/fs

plt.figure()
plt.plot(t,x)
plt.figure()
plt.plot(tau,Rxx)


noise = np.random.uniform(size=x.size)
xn = x+noise 
tau, Rxx = xcorr(xn,xn, fs)
plt.figure()
plt.plot(t,xn)
plt.figure()
plt.plot(tau,Rxx)
# %% 8.4
import numpy as np

A = 1
B = 1
C = 2
D = 2
thetax = 0
thetay = -np.pi/4
phi = np.pi/2
n = 2
w = 2 * np.pi * 1
fs = 200
T = 100
t = np.linspace(0, T, T*fs)
rel_time_delay = (thetax - thetay) / w

x = A * np.sin(w * t + thetax) + B
y = C * np.sin(w * t + thetay) + D * np.sin(n * w * t + phi)

plt.figure()
plt.plot(t,x)
plt.plot(t,y)
plt.xlim(0,4)

tau, Rxx = xcorr(x,y,fs)
plt.figure()
plt.plot(tau,Rxx)
plt.xlim(-4,4)
plt.ylim(-1,1)
# %% 8.5

A=1
w = 2*pi*1 
fs=200
T=100
t=np.linspace(0,T, T*fs, endpoint=False)
s=A*np.sin(w*t)
n = 2*A*np.random.uniform(-1,1,size=s.size)
fc = 20

b, a = butter(9, fc/(fs/2))

y = s+n
tau, Ryy = xcorr(y,y, fs)

plt.figure()
plt.plot(t,y)
plt.plot(t,s,'--')
plt.xlim(0,4)
plt.figure()
plt.plot(tau,Ryy)
plt.xlim(0,4)
plt.ylim(-1,1)
# %% 8.6



fs = 1000
T = 5
t = np.arange(0, T, 1/fs)
np.random.seed(0)
s = np.random.randn(len(t))

fc = 100
b, a = butter(9, fc/(fs/2))
s = lfilter(b, a, s)
s = s - np.mean(s)
s = s / np.std(s)
delta = 0.2
x = s[int(delta*fs):]
y = s[:-int(delta*fs)]
np.random.seed(1)
nx = np.random.randn(len(x)) * np.std(s)
np.random.seed(2)
ny = np.random.randn(len(y)) * np.std(s)
x = x + nx
y = y + ny

tau, Rxy = xcorr(x, y, fs)
tau2, Rxx = xcorr(x, x, fs)

plt.figure()

plt.plot(tau2, Rxx)
plt.xlabel(r'Time delay (s)')
plt.ylabel(r'Auto-correlation')
plt.xlim(-0.25,0.25)

plt.figure()
plt.plot(tau, Rxy)
plt.xlabel(r'Time delay (s)')
plt.ylabel(r'Cross-correlation')
plt.xlim(-0.5,0.5)
plt.ylim(-1,1)

# %% 8.7


fs = 200
t = np.arange(0, 1, 1/fs)
x = chirp(t, f0=5, t1=1, f1=15)
h = np.flip(x)
plt.figure()
plt.plot(t,x)
plt.figure()
plt.plot(t,h)


delta = 2
a = 0.1
fs = 200  
y = np.concatenate((np.zeros(delta*fs), a*x, np.zeros(3*fs)))
t = np.arange(len(y)) / fs

np.random.seed(0)
noise = 2 * np.std(a*x) * np.random.randn(len(y))


y = y + noise

plt.figure()

plt.plot(t, y)

tau, Rxy = xcorr(y,x,fs)
plt.figure()
plt.plot(tau, Rxy)
plt.xlim(-1,5)
#Ryx = correlate(y,x,'full')
#tau = correlation_lags(y.size, x.size, mode='full')
tau, Ryx = xcorr(y,x,fs)

plt.figure()

plt.plot(tau, Ryx)
plt.xlim(-1,5)



out = convolve(y, h, mode='full')


plt.figure()

plt.plot(t, out[:y.size])
# %% 8.8

A = 1
C = 2
D = 2
thetax = 0
thetay = -np.pi/4
phi = np.pi/2
n = 2
p = 1
w = 2*np.pi*p
fs = 200
T = 100
t = np.linspace(0,T,fs*T)
x = A*np.sin(w*t+thetax)
y = C*np.sin(w*t+thetay)+D*np.sin(n*w*t+phi)
maxlag = 4*fs

plt.figure()
plt.plot(t,x)
plt.plot(t,y)
plt.xlim(0,4)

tau, Rxy = xcorr(x, y,fs)

plt.figure()
plt.plot(tau, Rxy)
plt.xlabel('Time delay (s)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between x and y')
plt.xlim(0,4)

plt.show()



ii = np.where(tau==0)[0][0]
Sxy = fft(Rxy[ii:])
f = fftfreq(Sxy.size, 1/fs)[0:Sxy.size//2]
Sxy = Sxy[0:Sxy.size//2]
thetaxy=(thetax-thetay)


plt.figure()
plt.plot(f,np.abs(Sxy))
plt.xlim(0,2*p)
ind = np.where(f == p)[0][0]
arg_Sxy_at_p_Hz = np.angle(Sxy[ind])
print(thetaxy)
print(arg_Sxy_at_p_Hz)


# %% 8.9
fs=500
T=100
t = np.linspace(0,T,T*fs)
randgen = np.random.default_rng(seed=0)
s=randgen.random(t.size)
plt.figure()
plt.plot(t,s)
fc=100
b,a=butter(22,fc/(fs/2))
s=filtfilt(b,a,s)
plt.figure()
plt.plot(t,s)
s=s-np.mean(s)
s=s/np.std(s)# Makes mean(s)=0 & std(s)=1
delta=0.2
plt.figure()
plt.plot(t,s)



x=s
y = np.zeros_like(x)
y[int(delta*fs):]=s[:int(-delta*fs)]

plt.figure()
plt.plot(x)
plt.plot(y,'--')


nx=randgen.random(x.size)
ny=randgen.random(y.size)
x=x+nx
y=y+ny
tau1, Rxx=xcorr(x,x,fs)
tau2, Rxy=xcorr(y,x,fs)

plt.figure()
plt.plot(tau1,Rxx)
plt.plot(tau2,Rxy)
plt.xlim(-2,2)
plt.ylim(0,1.5)

ii = np.where(tau2==0)[0][0]
Sxy = fft(Rxy[ii:ii+fs])
#Sxy /= Sxy.size
f2 = fftfreq(Sxy.size, 1/fs)[0:Sxy.size//2]
Sxy = Sxy[0:Sxy.size//2]
plt.figure()
plt.plot(f2,np.unwrap(np.angle(Sxy)))

ii = np.where(tau1==0)[0][0]
Sxx = fft(Rxx[ii:ii+fs])
#Sxx /= Sxx.size
f1 = fftfreq(Sxx.size, 1/fs)[0:Sxx.size//2]
Sxx = Sxx[0:Sxx.size//2]
H1 = Sxy/Sxx




plt.figure()
plt.plot(f2,20*np.log10(np.abs(Sxy)))
plt.plot(f1,20*np.log10(np.abs(H1)))
plt.figure()
plt.plot(f2,np.unwrap(np.angle(Sxy)))
plt.plot(f1,np.unwrap(np.angle(H1)))

# %%8.10
fs=100

t=np.linspace(0,int(2.5),int(2.5*fs))
A=100
zeta=0.03
f=10
wn=2*pi*f; wd=np.sqrt(1-zeta**2)*wn
h=(A/wd)*np.exp(-zeta*wn*t)*np.sin(wd*t)

T=100 # 100 and 2000 
x=2*randgen.random(T*fs)
y=np.convolve(h,x)
y=lfilter(h,1,x)
tau, Rxx = xcorr(x,x,fs)
tau, Ryy = xcorr(y,y,fs)
tau, Rxy = xcorr(y,x,fs)

plt.figure()
plt.plot(tau,Rxx)
plt.xlim(-5,5)
plt.figure()
plt.plot(tau,Ryy)
plt.xlim(-5,5)
plt.figure()
plt.plot(tau,Rxy)
plt.xlim(-5,5)

ii = np.where(tau==0)[0][0]
Sxx = fft(Rxx[ii:ii+fs])
#Sxx = fft(Rxx[fs:6*fs])[0:5*fs//2-1]
f = fftfreq(Sxx.size, 1/fs)[:Sxx.size//2]
Sxx = Sxx[:Sxx.size//2]
ii = np.where(tau==0)[0][0]
Syy = fft(Ryy[ii:ii+fs])
#Syy = fft(Ryy[0:5*fs])[0:5*fs//2-1]
Syy = Syy[:Syy.size//2]
ii = np.where(tau==0)[0][0]
Sxy = fft(Rxy[ii:ii+fs])
#Sxy = fft(Rxy[fs:6*fs])[0:5*fs//2-1]
Sxy = Sxy[:Sxy.size//2]

plt.figure()
plt.plot(f,20*np.log10(np.abs(Sxx)))
#plt.xlim(-5,5)
plt.figure()
plt.plot(f,20*np.log10(np.abs(Syy)))
#plt.xlim(-5,5)
plt.figure()
plt.plot(f,20*np.log10(np.abs(Sxy)))
#plt.xlim(-5,5)



H1 = Sxy/Sxx
H = fftshift(fft(h))
ff = fftshift(fftfreq(H.size,1/fs))
plt.figure()
plt.plot(f,20*np.log10(np.abs(H1)))
#plt.xlim(-5,5)
plt.figure()
plt.plot(ff,20*np.log10(np.abs(H)))
# %%9.1

fs = 500
T1 = 1
T2 = 100
t1 = np.arange(0, T1, 1/fs)
t2 = np.arange(0, T2, 1/fs)
T = 0.1
h = 1/T * np.exp(-t1/T)

np.random.seed(0)
x = np.random.randn(T2*fs)
fc = 30
b, a = butter(9, fc/(fs/2))
x = lfilter(b, a, x)
x = x - np.mean(x)
x = x / np.std(x)
y = np.convolve(h, x, mode='full')
#y = sg.convolve(y, x, mode='full')

N = 4*fs
fxx, Sxx = csd(x, x, window='han', nperseg=N, noverlap=N//2, fs=fs, scaling='spectrum', return_onesided=False)
fyy, Syy = csd(y, y, window='han', nperseg=N, noverlap=N//2, fs=fs, scaling='spectrum', return_onesided=False)
fxy, Sxy = csd(x, y, window='han', nperseg=N, noverlap=N//2, fs=fs, scaling='spectrum', return_onesided=False)

Sxx = np.fft.fftshift(Sxx)
Syy = np.fft.fftshift(Syy)
Sxy = np.fft.fftshift(Sxy)

fxx = np.fft.fftshift(fxx)
fyy = np.fft.fftshift(fyy)
fxy = np.fft.fftshift(fxy)

H1 = Sxy / Sxx
H = np.fft.fftshift(np.fft.fft(h, N))
#H = np.fft.fft(h, N)/ fs
Gamma = np.abs(Sxy)**2 / (Sxx * Syy)
f = np.fft.fftshift(fftfreq(H.size,1/fs))



plt.figure(1)
plt.plot(fxx, 10*np.log10(np.abs(Sxx)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sxx (dB)')
plt.axis([-30, 30, -30, -15])

plt.figure(2)
plt.plot(fxy, 10*np.log10(np.abs(Sxy)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Sxy| (dB)')
plt.axis([-30, 30, -15, 5])


plt.figure(3)
plt.plot(f, 10*np.log10(np.abs(H1)))
plt.plot(f, 10*np.log10(np.abs(H)), 'r:')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|H1| (dB)')
plt.xlim(-30,30)
plt.axis([-30, 30, 15, 30])

plt.figure(4)
plt.plot(f, np.unwrap(np.angle(H1)))
plt.plot(f, np.unwrap(np.angle(H)), 'r:')
plt.xlabel('Frequency (Hz)')
plt.ylabel('arg H1 (rad)')
plt.xlim(-30,30)
#plt.axis([-30, 30, -1.6, 1.6])

plt.figure(5)
plt.plot(f, Gamma)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence function')
plt.xlim(-150,150)
#plt.axis([-150, 150, 0, 1.1])
# %% 9.2
fs=500; T1=1; T2=40
t1 = np.arange(0,T1,1/fs)
t2 = np.arange(0,T2,1/fs)
T=0.1
h=1/T*np.exp(-t1/T)
A=2
f=1
w=2*pi*f
x=A*np.sin(w*t2)
y=lfilter(h,1,x)/fs

maxlag=2*fs
tau, Ryy=xcorr(y,y,fs)
tau, Rxy=xcorr(y,x,fs)
phi=np.atan(1/(w*T))
Ryy_a=(A**2/2)*(1./(1+(w*T)**2))*np.cos(w*tau)
Rxy_a=(A**2/2)*(1./np.sqrt(1+(w*T)**2))*np.sin(w*tau+phi)

plt.figure(1) 
plt.plot(tau,Ryy)
plt.plot(tau,Ryy_a, 'r:') 
plt.xlim(-2,2)


plt.figure(2) 
plt.plot(tau,Rxy) 
plt.plot(tau,Rxy_a, 'r:') 
plt.xlim(-2,2)

# %% 9.3


fs = 100
T = 500
t = np.arange(0, T-1/fs, 1/fs)
np.random.seed(0)
s = np.random.randn(len(t))
fc = 10
b, a = butter(9, fc/(fs/2))
s = filtfilt(b, a, s)
s = s - np.mean(s)
s = s / np.std(s)
a = 1
b = 0.8
c = 0.75
delta1 = 1
delta2 = 1.5
N1 = 2*fs
N2 = T*fs - N1
x = a * s
y = np.zeros_like(x)
y[int(delta1*fs):] += b * s[:int(-delta1*fs)]
y[int(delta2*fs):] += c * s[:int(-delta2*fs)]


np.random.seed(10)
n = np.random.randn(len(y)) * 0.1
y = y + n
maxlag = 2*fs
tau, Rxy = xcorr(y, x, fs)

T1 = 50
f, Gxx  = csd(x, x, window='han', nperseg=T1*fs, noverlap=T1*fs//2, fs=fs, scaling='spectrum')
f, Gyy  = csd(y, y, window='han', nperseg=T1*fs, noverlap=T1*fs//2, fs=fs, scaling='spectrum')
f, Gxy  = csd(x, y, window='han', nperseg=T1*fs, noverlap=T1*fs//2, fs=fs, scaling='spectrum')
Gamma = np.abs(Gxy)**2 / (Gxx * Gyy)


plt.figure(1)
plt.plot(tau[maxlag+1:], Rxy[maxlag+1:])
plt.xlabel('Lag (τ)')
plt.ylabel('Cross-correlation')
plt.axis([0, 2, -0.2, 0.8])

plt.figure(2)
plt.plot(f, np.unwrap(np.angle(Gxy)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('arg Gxy (rad)')
plt.axis([0, 15, -120, 10])

plt.figure(3)
plt.plot(f, Gamma)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence function')
plt.axis([0, 15, 0, 1.2])


# %% 9.4


# Constants
A1 = 20
A2 = 30
f1 = 5
f2 = 15
wn1 = 2 * np.pi * f1
wn2 = 2 * np.pi * f2
zeta1 = 0.05
zeta2 = 0.03
wd1 = np.sqrt(1 - zeta1**2) * wn1
wd2 = np.sqrt(1 - zeta2**2) * wn2
fs = 50
T1 = 10
t1 = np.arange(0, T1, 1/fs)

# Impulse response
h = (A1 / wd1) * np.exp(-zeta1 * wn1 * t1) * np.sin(wd1 * t1) + (A2 / wd2) * np.exp(-zeta2 * wn2 * t1) * np.sin(wd2 * t1)

# Signal
T = 5000
np.random.seed(0)
x = np.random.randn(T * fs)
y = lfilter(h, 1, x)

# Noise
np.random.seed(10)
nx = 0.5 * np.random.randn(len(x))  
nx = 0 #for Case (a)
np.random.seed(20)
ny = 0.5 * np.random.randn(len(y))  
#ny = 0 #for Case (b)
x = x + nx
y = y + ny


# Power spectral density
N1 = T1 * fs
f, Gxx = csd(x, x, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum')
f, Gyy = csd(y, y, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum')
f, Gxy = csd(x, y, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum')
f, Gyx = csd(y, x, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum')

# Frequency response
H1 = Gxy / Gxx
H2 = Gyy / Gxy
HT = (Gyy - Gxx + np.sqrt((Gxx - Gyy)**2 + 4 * np.abs(Gxy)**2)) / (2 * Gxy)
H = fft(h)
fh = fftfreq(H.size,1/fs)[:H.size//2]
H = H[:H.size//2]

# Plots

plt.figure(1)
plt.plot(f, 20 * np.log10(np.abs(H1)), fh, 20 * np.log10(np.abs(H)), 'r:')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|H1(f)| (dB)')
plt.axis([0, 25, -30, 25])

plt.figure(2)
plt.plot(f, 20 * np.log10(np.abs(H2)), fh, 20 * np.log10(np.abs(H)), 'r:')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|H2(f)| (dB)')
plt.axis([0, 25, -30, 25])

plt.figure(3)
plt.plot(f, 20 * np.log10(np.abs(HT)), fh, 20 * np.log10(np.abs(H)), 'r:')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|HT(f)| (dB)')
plt.axis([0, 25, -30, 25])

plt.show()

# %% 9.5

mat = loadmat('beam_experiment.mat')

# Constants
fs = 256
T = 4
# Generate signals x and y

x = mat['x'][0]
y = mat['y'][0]

# Compute power spectral density (PSD)

N1 = T*fs

f, Gxx = csd(x, x, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum', return_onesided=True)
f, Gyy = csd(y, y, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum', return_onesided=True)
f, Gxy = csd(x, y, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum', return_onesided=True)
f, Gyx = csd(y, x, window='han', nperseg=N1, noverlap=N1 // 2, fs=fs, scaling='spectrum', return_onesided=True)

# Compute frequency response
H1 = Gxy / Gxx
H2 = Gyy / Gyx
HT = (Gyy - Gxx + np.sqrt((Gxx - Gyy) ** 2 + 4 * np.abs(Gxy) ** 2)) / (2 * Gyx)

# Plot frequency response


#plt.figure(1)
#plt.plot(f, 20 * np.log10(np.abs(H1)))
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('|$H_1(f)|$ (dB)')
#plt.axis([5, 90, -45, 25])
#
#plt.figure(2)
#plt.plot(f, 20 * np.log10(np.abs(H2)))
#
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('|$H_2(f)|$ (dB)')
#plt.axis([5, 90, -45, 25])
#
#plt.figure(3)
#plt.plot(f, 20 * np.log10(np.abs(HT)))
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('|$H_T(f)|$ (dB)')
#plt.axis([5, 90, -45, 25])

plt.figure(4)
plt.plot(f, 20 * np.log10(np.abs(H1)))
plt.plot(f, 20 * np.log10(np.abs(H2)), '--',)
plt.plot(f, 20 * np.log10(np.abs(HT)), ':')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|$H(f) Estimates|$ (dB)')
plt.legend([r'$H_1$', r'$H_2$', r'$H_T$'],loc='upper right')
plt.axis([5, 90, -45, 25])

plt.figure(5)
plt.plot(f, np.unwrap(np.angle(H1)), 
         f, np.unwrap(np.angle(H2)), '--',
         f, np.unwrap(np.angle(HT)), ':')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase spectrum (rad)')
plt.legend([r'$H_1$', r'$H_2$', r'$H_T$'],loc='upper right')
plt.axis([5, 90, -7, 0.5])

plt.show()
# %%
