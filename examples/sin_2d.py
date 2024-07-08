"""Learn 2D sine transform
"""
#%%
import jax 
import optax
import equinox as eqx
import jax.numpy as jnp 
import jax.random as jr 
import numpy as np
import interpax
import diffrax as dfx
import matplotlib.pyplot as plt
import deeponetx.data.function_spaces as fs 
import deeponetx.data.kernels as kernels
import deeponetx.nn as nn
from deeponetx.data.data import DataDeepONet
import deeponetx.train as traindtx


key = jr.PRNGKey(0)
k_branch, k_trunk, k_bias, k_data = jr.split(key, num=4)

m = 70 # resolution
n_samp = 1150 # number of samples
# Sensor points
x = jnp.linspace(0,10,num=m)
y = jnp.linspace(0,10,num=len(x))
xv, yv = jnp.meshgrid(x,y)
X = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
k = kernels.SquaredExponential(10., 1.0)
grf = fs.GaussianRandomField(k, X, jitter=1e-3)
us = grf.sample(n_samp, key=key)

#%%
us_fn = interpax.Interpolator2D(x, y, xv, yv, us[0,:].reshape(m,m))

#%%

# Compute 2D sine transform
def transform(fn, x, y):
    return jnp.sin(fn(x+y))

data = DataDeepONet(us, X, jax.vmap(transform, in_axes=(0,0))(X[:,0], X[:,1])) # [, 1D], [, 2D], [,1D]
data_train, data_test = data[:800], data[800:]

# Setting DeepOnet
interacti_size = 100
net_branch = eqx.nn.MLP(
    in_size=data.input_branch.shape[1], 
    out_size= interacti_size,
    width_size = 100,
    depth=1,
    activation=jax.nn.relu,
    key= k_branch
)
net_trunk = eqx.nn.MLP(
    in_size=2, 
    out_size= interacti_size,
    width_size =100,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.relu,
    key= k_trunk
)

bias = jax.random.uniform(k_bias, shape=(1,))

net = nn.UnstackDeepONet(net_branch, net_trunk, bias)

# Training
netopt, losses = traindtx.train(net, data_train, optax.adam(1e-3), 10)
# %% Validate 
def visualize(net, data, i=0):
    x = jnp.linspace(0,10,num=100)
    y = jnp.linspace(0,10,num=100)
    xv, yv = jnp.meshgrid(x,y)
    X = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
    
    fig, ax = plt.subplots()
    ax.imshow(input_d)
    return fig, ax


visualize(net, data_test, i=1)

# %%
