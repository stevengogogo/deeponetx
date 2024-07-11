"""
Inspired from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/blob/main/Diffusion-reaction/DeepONet_DR.ipynb
"""
#%%
from deeponetx.data import data as dtxdata
import deeponetx.train as traindtx
from deeponetx import nn
import numpy as np
from jax import random
import jax.numpy as jnp
from jax import lax
import jax.random as jr
from jax import config
import equinox as eqx
import jax 
import optax
import equinox as eqx
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", False)

# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

# A diffusion-reaction numerical solver
def solve_ADR(key, Nx, Nt, P, length_scale):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01*np.ones_like(x)
    v = lambda x: jnp.zeros_like(x)
    g = lambda u: 0.01*u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: jnp.zeros_like(x)

    # Generate subkeys
    subkeys = random.split(key, 2)

    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (N,)))
    # Create a callable interpolation function  
    f_fn = lambda x: jnp.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = jnp.eye(Nx, k=1) - jnp.eye(Nx, k=-1)
    D2 = -2 * jnp.eye(Nx) + jnp.eye(Nx, k=-1) + jnp.eye(Nx, k=1)
    D3 = jnp.eye(Nx - 2)
    M = -jnp.diag(D1 @ k) @ D1 - 4 * jnp.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * jnp.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * jnp.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx, Nt))
    u = u.at[:,0].set(u0(x))
    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = jnp.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u = u.at[1:-1, i + 1].set(jnp.linalg.solve(A, b1 + b2))
        return u
    # Run loop
    UU = lax.fori_loop(0, Nt-1, body_fn, u)

    # Input sensor locations and measurements
    xx = jnp.linspace(xmin, xmax, m)
    u = f_fn(xx)
    # Output sensor locations and measurements
    idx = random.randint(subkeys[1], (P, 2), 0, max(Nx, Nt))
    y = jnp.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
    s = UU[idx[:,0], idx[:,1]]
    # x, t: sampled points on grid
    return (x, t, UU), (u, y, s)

def generate_one_training_data(key, P):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx , Nt, P, length_scale)

    u = jnp.tile(u, (P, 1))

    return u, y, s

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    u_train, y_train, s_train= jax.vmap(generate_one_training_data, (0, None))(keys, P)

    u_train = jnp.float32(u_train.reshape(N * P, -1))
    y_train = jnp.float32(y_train.reshape(N * P, -1))
    s_train = jnp.float32(s_train.reshape(N * P, -1))

    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train

key = random.PRNGKey(0)
k_trunk, k_branch, k_bias, k_batch = random.split(key, 4)

# GRF length scale
length_scale = 0.2

# Resolution of the solution
Nx = 100
Nt = 100

N = 100 # number of input samples
m = Nx   # number of input sensors
P_train = 100 # number of output sensors
n_batch = 1_0000

u_train, y_train, s_train = generate_training_data(key, N, P_train)



data = dtxdata.DatasetDeepONet(u_train, y_train, s_train, n_batch, key=k_batch)

#%%

net_branch = eqx.nn.MLP(
    in_size=m, 
    out_size= 50,
    width_size =50,
    depth=4,
    activation=jax.nn.tanh,
    key= k_branch)
net_trunk = eqx.nn.MLP(
    in_size=2, 
    out_size= 50,
    width_size =50,
    depth=4,
    activation=jax.nn.tanh,
    key= k_trunk)


net = nn.UnstackDeepONet(net_branch, net_trunk)

# %%
# Training
lr = optax.exponential_decay(1e-2, 2000, 0.9)
opt = optax.adam(lr)
netopt, losses = traindtx.train(net, data, opt, 120000)

# %%
