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
from typing import NamedTuple
import sys 
import os
sys.path.append('.')
import deeponetx


class Data(NamedTuple):
    us: jnp.ndarray # conditioned function (sampled) [nsample, ngrid]
    ys: jnp.ndarray # location y [nsample, locations]
    Guys: jnp.array # output of operator at location y [nsample, locations]
    def __getitem__(self, key):
        """Get subset of data
        """
        return Data(self.us[key], self.ys[key], self.Guys[key])

def sample_grf1d(m, tspan, *, key:int):
    model = Gaussian(dim=1, var=1, len_scale=10.)
    ts = np.linspace(tspan[0], tspan[1], m)
    srf = SRF(model, seed=key)
    vs = srf.structured(range(m))
    return vs
    #Vfn = dfx.LinearInterpolation(ts, vs)
    #return Vfn.evaluate

def get_data():
    """Create align data
    """
    m = 100 # resolution
    tspan = [0.,1.]
    n_samp = 1150

    Vs = jnp.array([sample_grf1d(m, tspan, key=i) for i in range(n_samp)])

    def solve(vs):
        """Solve ODE for given GRF function
        """
        # Observe points
        v0 = 0. 
        ts = jnp.linspace(tspan[0], tspan[1], len(vs))
        # interpolation
        vfn = dfx.LinearInterpolation(ts, vs).evaluate

        # ODE setting 
        def ode(t, v, args):
            return vfn(t)
        
        # solve setting
        term = dfx.ODETerm(ode)
        solver = dfx.Dopri5()
        # solve
        sol = dfx.diffeqsolve(term, solver, t0=tspan[0], t1=tspan[1], dt0=(tspan[1]-tspan[0])/m, y0=v0, saveat=dfx.SaveAt(ts=ts))

        vs = vfn(ts)

        return sol, vs
    # Solve on VS
    sols, vs = jax.vmap(solve, in_axes=(0,))(Vs)
    return Data(
        us = vs, # conditioned functions
        ys = sols.ts, # location y
        Guys = sols.ys # output of operator at location y
    )
# %%
def main():

    data = get_data()

    data_train = data[:150]
    data_test = data[150:]
    