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
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

key = jr.PRNGKey(0)
k_branch, k_trunk, k_bias, k_data = jr.split(key, num=4)

m = 100 # resolution
n_samp = 1000 # number of samples
# Sensor points
x = jnp.linspace(0,2*jnp.pi,num=m)
y = jnp.linspace(0,2*jnp.pi,num=len(x))
xv, yv = jnp.meshgrid(x,y)
X = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
k = kernels.SquaredExponential(1.0, 1.) 
grf = fs.GaussianRandomField(k, X, jitter=1e-3)
us = grf.sample(n_samp, key=key)

# Compute 2D sine transform
def operator(x):
    return x

def transform(us, Xs):
    us_fn = interpax.Interpolator2D(x, y, us.reshape(m,m))
    u = us_fn(Xs[:,0], Xs[:,1]) # value at locations
    return operator(u)

data = DataDeepONet(us.reshape(n_samp,1,m,m), X.reshape(-1,2), jax.vmap(transform, in_axes=(0, None))(us, X)) # [, 1D], [, 2D], [,1D]
data_train, data_test = data[: n_samp //2], data[n_samp//2:]
#%%

# Setting DeepOnet
interacti_size = 300
class CNN(eqx.Module):
    layers: list 
    def __init__(self, key):
        key1, key2, key3, key4 = jr.split(key, num=4)
        self.layers= [
            eqx.nn.Conv2d(1,3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu, 
            jnp.ravel,
            eqx.nn.Linear(27648, 512, key=key2),
            jax.nn.tanh,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu, 
            eqx.nn.Linear(64, interacti_size,key= key4)
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

net_branch = CNN(k_branch)
net_trunk = eqx.nn.MLP(
    in_size=2, 
    out_size= interacti_size,
    width_size =100,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.relu,
    key= k_trunk)

bias = jax.random.uniform(k_bias, shape=(1,))

net = nn.UnstackDeepONet(net_branch, net_trunk, bias)

# Training
netopt, losses = traindtx.train(net, data_train, optax.adam(1e-2), 10, batch_size=n_samp//2, key=k_data)
# %% Validate 
def visualize(data):
    fig, ax = plt.subplots()
    ax.pcolormesh(data.input_trunk[:,0].reshape(m,m), data.input_trunk[:,1].reshape(m,m), data.output.reshape(m,m), shading='auto')
    return fig, ax

data_pred =  DataDeepONet(data_test.input_branch, 
                          data_test.input_trunk, 
                          traindtx.predict(netopt, data_test).reshape(-1, m,m))

fig, ax = visualize(data_test[1])
fig2, ax2 = visualize(data_pred[1])
ax.set_title("Truth")
ax2.set_title("Prediction")

# %%
