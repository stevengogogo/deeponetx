"""
Learning explicit operator

Data generation: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
"""
#%%
import jax
import jax.numpy as jnp 
import jax.random as jr
import numpy as np
import diffrax as dfx
import matplotlib.pyplot as plt
from gstools import SRF, Gaussian
import optax
import deeponetx as dtx
from deeponetx import nn
from deeponetx import train as traindtx
from deeponetx.data import function_spaces, kernels

def sample_grf1d(n_samp, ts, key:int):
    kernel = kernels.SquaredExponential(length_scale=0.1, signal_stddev=1.)
    model = function_spaces.GaussianRandomField(kernel, ts)
    vs = model.sample(n_samp, key=key)
    print(vs.shape)
    return vs

def get_data(key=0):
    """Create align data
    """
    m = 100 # resolution
    ts = jnp.linspace(0, 1, num=m)
    n_samp = 1150
    key = jr.PRNGKey(0)
    Vs = sample_grf1d(n_samp, ts, key=key)

    def solve(vs):
        """Solve ODE for given GRF function
        """
        # Observe points
        v0 = 0. 
        # interpolation
        vfn = dfx.LinearInterpolation(ts, vs).evaluate

        # ODE setting 
        def ode(t, v, args):
            return vfn(t)
        
        # solve setting
        term = dfx.ODETerm(ode)
        solver = dfx.Dopri5()
        # solve
        sol = dfx.diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=(ts[1]-ts[0]), y0=v0, saveat=dfx.SaveAt(ts=ts))

        vs = vfn(ts)

        return sol, vs
    # Solve on VS
    sols, vs = jax.vmap(solve, in_axes=(0,))(Vs)
    ys = sols.ts[0].reshape(-1, 1)
    return traindtx.DataDeepONet(vs, ys, sols.ys)

def visualize(net:dtx.nn.AbstractDeepONet, data:dtx.train.DataDeepONet, i=0):
    fig, ax = plt.subplots()
    ax.plot(data.input_trunk[:,0], data.input_branch[i,:], label="input")
    ax.plot(data.input_trunk[:,i], data.output[i,:], label="Antiderivative (truth)" )
    ax.plot(data.input_trunk[:,0], 
        jax.vmap(
            net, in_axes=(None, 0)
        )(data.input_branch[i,:], data.input_trunk),
        label="Prediction"        
    )
    ax.legend()
    return fig, ax

def vis_loss(losses:list):
    fig, ax = plt.subplots()
    ax.plot(losses, label="Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Traing Loss (MSE)")
    ax.set_title("Training Loss")
    return fig, ax


def main():
    # Create key
    key = jr.PRNGKey(0)

    # Create dta
    data = get_data()
    data_train = data[:500]
    data_test = data[500:]

    # net setting
    width_size = 40 
    depth = 1
    interact_size = 40
    activation = jax.nn.relu
    optimizer = optax.adam(1e-3)

    # Create net
    in_size_branch = data.input_branch.shape[1]
    net = dtx.nn.create_UnstackDeepONet1d_MLP(in_size_branch, width_size, depth, interact_size, activation, key=key)

    # Training
    net, losses = traindtx.train(net, data_train, optimizer, 10000)

    # visualize
    fig, ax = visualize(net, data_test, i = 0)
    fig2, ax2 = visualize(net, data_test, i = 1)
    fig3, ax3 = vis_loss(losses)
    ax.set_title("Test 0")

main()




# %%
