"""
Learning explicit operator

Data generation: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
"""
#%%
import jax
import jax.numpy as jnp 
import numpy as np
import diffrax as dfx
import matplotlib.pyplot as plt
from gstools import SRF, Gaussian
import sys 
import os
sys.path.append('.')
import deeponetx

def sample_grf1d(m, tspan, *, key:int):
    model = Gaussian(dim=1, var=1, len_scale=10.)
    ts = np.linspace(tspan[0], tspan[1], m)
    srf = SRF(model, seed=key)
    vs = srf.structured(range(m))
    Vfn = dfx.LinearInterpolation(ts, vs)
    return Vfn.evaluate

def ode(t, v, args):
    return args(t)

def data():
    """Create align data
    """
    m = 100 # resolution
    xspan = [0.,1.]
    n_samp = 1150

    Vs = [sample_grf1d(m, xspan, key=i) for i in range(n_samp)]

    # Solve on VS
    v0 = 0.
    term = dfx.ODETerm(ode)
    solver = dfx.Dopri5()
    solve = lambda args: dfx.diffeqsolve(term, solver, t0=xspan[0], t1=xspan[1], dt0=(xspan[1]-xspan[0])/m, y0=v0, args=args)

    solutions = [solve(Vs[i]) for i in range(n_samp)]
    
    return solutions
# %%

## Interpolation
# Sample code for interpolation
t = jnp.linspace(0, 2*jnp.pi, 100)
y = jnp.sin(t)
f = dfx.LinearInterpolation(t, y)

plt.plot(t, f.evaluate(t))
