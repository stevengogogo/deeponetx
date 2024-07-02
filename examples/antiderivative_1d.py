"""
Learning explicit operator

Data generation: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
"""
#%%
import jax.numpy as jnp 
import numpy as np
import diffrax as dfx
import matplotlib.pyplot as plt
from gstools import SRF, Gaussian
import sys 
import os
sys.path.append('.')
import deeponetx

def sample_grf1d(n_samp, m):
    model = Gaussian(dim=1, var=1, len_scale=10.)
    Vs = np.zeros((n_samp, m))
    for i in range(n_samp):
        srf = SRF(model, seed=i)
        vs = srf.structured(range(m))
        Vs[i,:] = vs
    return Vs

def ode(t, v, args):
    return args(t)

def data():
    """Create align data
    """
    m = 100 # resolution
    xspan = [0.,1.]
    n_samp = 1150

    Vs = sample_grf1d(n_samp, m)

    # Solve on VS
    Us = np.zeros((n_samp, m))


    def vfn(t):
        return 0.

    v0 = 0.
    term = dfx.ODETerm(ode)
    solver = dfx.Dopri5()
    solution = dfx.diffeqsolve(term, solver, t0=xspan[0], t1=xspan[1], dt0=(xspan[1]-xspan[0])/m, y0=v0)
    
# %%
