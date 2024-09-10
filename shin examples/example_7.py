# %%
import numpy as np
import matplotlib.pyplot as plt

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

x = np.random.uniform(size=(10,5000))
S = np.sum(x, axis=-1)
nbin=20
N = x.size
h, s = np.histogram(x,bins=nbin)

plt.hist(x, bins=nbin, axis=-1)
# %%
