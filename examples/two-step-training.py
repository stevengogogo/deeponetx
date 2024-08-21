"""
Perform two-step training on the antiderivative dataset.
- https://arxiv.org/abs/2309.01020
"""

#%%
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
import wget
import numpy as np
from deeponetx.data.data import DataDeepONet
from deeponetx import nn as dnn

# Data 
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
from deeponetx.data.data import DatasetDeepONet, DataDeepONet

def sample_grf1d(n_samp, ts, key:int):
    kernel = kernels.SquaredExponential(length_scale=0.1, signal_stddev=1.)
    model = function_spaces.GaussianRandomField(kernel, ts)
    vs = model.sample(n_samp, key=key)
    print(vs.shape)
    return vs

def get_data(key:jr.PRNGKey, n_samp, m, train_size):
    """Create align data
    - n_samp: number of samples
    - m: number of sensor points
    - train_size: size of training data
    """
    k_d1, k_d2, k_s, k_c = jr.split(key, num=4)
    ts = jnp.linspace(0, 1, num=m) # sensor points
    Vs = sample_grf1d(n_samp, ts, key=k_s)

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
    gu = sols.ys
    ys = sols.ts[0].reshape(-1, 1)

    # Reshaping for betching 
    data_test = DataDeepONet(vs[train_size:], ys, gu[train_size:])
    data_train = DataDeepONet(vs[:train_size], ys, gu[:train_size])
    return data_train, data_test

# Traing 

    
@eqx.filter_jit
def update_fn(model, loss_fn, data:DataDeepONet, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, data)
    #jax.debug.print("{w}", w=grad.net_branch.layers[0].weight[0])
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss

def train_step_1(net_trunk, A, data, optimizer, n_iter):
    def loss_fn(netA, data):
        """|\psi A - U|"""
        net_trunk, A = netA
        out_trunk = jax.vmap(net_trunk)(data.input_trunk)
        out_trunk_reparam = out_trunk @ A
        errs = out_trunk_reparam - data.output.T
        mse = jnp.mean(jnp.square(errs.flatten()))
        return mse
    
    netA = (net_trunk, A)
    state = optimizer.init(
        eqx.filter(netA, eqx.is_array)
    )
    losses = np.zeros(n_iter)

    with tqdm(range(n_iter)) as t:
        for i in t:
            netA, state, loss = update_fn(netA, loss_fn, data, optimizer, state)
            losses[i] = loss
            t.set_description(f'Loss: {loss}\t')
    net_trunk, A = netA
    return net_trunk, A, losses


def train_step_2(net_branch, net_trunk, A, data, optimizer, n_iter:int):
    psi = jax.vmap(net_trunk)(data.input_trunk)
    R = jnp.linalg.qr(psi, mode='r')
    RA = R@A # (N, K)

    def loss_fn(net_branch, data):
        """|\psi A - U|"""
        out_branch = jax.vmap(net_branch)(data.input_branch)
        errs = out_branch.T - RA
        mse = jnp.mean(jnp.square(errs.flatten()))
        return mse
    
    state = optimizer.init(eqx.filter(net_branch, eqx.is_array))
    losses = np.zeros(n_iter)

    with tqdm(range(n_iter)) as t:
        for i in t:
            net_branch, state, loss = update_fn(net_branch, loss_fn, data, optimizer, state)
            losses[i] = loss
            t.set_description(f'Loss: {loss}\t')
    T = jnp.linalg.inv(R)
    return net_branch, T, losses

def train_twostep(net_branch, net_trunk, A, data, optimizer, n_iter:int):
    net_trunk_opt, A_opt, losses1 = train_step_1(net_trunk, A, data, optimizer, n_iter)
    net_branch_opt, T, losses2 = train_step_2(net_branch, net_trunk_opt, A_opt, data, optimizer, n_iter)

    def deeponet(input_branch, input_trunk):
        out_branch = net_branch_opt(input_branch)
        out_trunk = net_trunk_opt(input_trunk)
        out_trunk_reparam = out_trunk @ T
        return jnp.sum(out_trunk_reparam * out_branch)

    return deeponet, (losses1, losses2)


#
key = jr.PRNGKey(0)
knet, key = jr.split(key)
# Get dataset 
K = 10000 # number of sampled function
my = 100 # number of realization points


n_train = K
n_test = 500
data_train, data_test = get_data(key, n_train+n_test, my, n_train)

# DeepONet
N = 40 # last layer size
assert my >= N
# Net 
k_trunk, k_branch, k_T = jr.split(knet, num=3)

net_branch = eqx.nn.MLP(
    in_size= data_train.input_branch.shape[1],
    out_size= N,
    width_size= 10,
    depth=1,
    activation=jax.nn.relu,
    key=k_branch
)

net_trunk = eqx.nn.MLP(
    in_size=data_train.input_trunk.shape[1],
    out_size=N,
    width_size=60,
    depth=3,
    activation=jax.nn.relu,
    key=k_trunk
)

A = jr.normal(k_T, shape=(N, K))

optimizer = optax.adam(1e-4)
deeponet, (loss1, loss2) = train_twostep(net_branch, net_trunk, A, data_train, optimizer, 100000)
#

# Validation

i = 5
plt.plot(
    data_test.input_trunk[:, 0],
    data_test.input_branch[i, :],
    label="input",
)
plt.plot(
    data_test.input_trunk[:, 0],
    data_test.output[i, :],
    label="gt antiderivative",
)
plt.plot(
    data_test.input_trunk[:, 0],
    jax.vmap(
        deeponet,
        in_axes=(None, 0)
    )(data_test.input_branch[i, :], data_test.input_trunk),
    label="prediction",
)
plt.legend()
plt.grid()
# 
def normalized_l2_error(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm

predictions_test = jax.vmap(
    jax.vmap(
        deeponet,
        in_axes=(None, 0)
    ),
    in_axes=(0, None,)
)(data_test.input_branch, data_test.input_trunk)

test_errors = jax.vmap(normalized_l2_error)(predictions_test, data_test.output)
mean_test_error = jnp.mean(test_errors)
std_test_error = jnp.std(test_errors)
print("Mean: {}; std: {}".format(mean_test_error, std_test_error))
# %%
