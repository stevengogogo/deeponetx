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
from tqdm.autonotebook import tqdm
import wget
import numpy as np
from deeponetx.data.data import DataDeepONet
from deeponetx import nn as dnn

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

    return net_branch, jnp.linalg.inv(R)




#
key = jr.PRNGKey(0)
knet, key = jr.split(key)
# Download dataset
wget.download("https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_train.npz", out="data")
wget.download("https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_test.npz", out="data")

# load data 
dataset_train = jnp.load("data/antiderivative_aligned_train.npz", allow_pickle=True)
branch_inputs_train = dataset_train["X"][0]
trunk_inputs_train = dataset_train["X"][1]
outputs_train = dataset_train["y"]

data = DataDeepONet(branch_inputs_train, trunk_inputs_train, outputs_train)

# Dim 
my = trunk_inputs_train.shape[0]
K = branch_inputs_train.shape[0]
N = 51

# Net 
k_trunk, k_branch, k_T = jr.split(knet, num=3)

net_branch = eqx.nn.MLP(
    in_size= branch_inputs_train.shape[1],
    out_size= N,
    width_size= 10,
    depth=3,
    activation=jax.nn.relu,
    key=k_branch
)

net_trunk = eqx.nn.MLP(
    in_size=trunk_inputs_train.shape[1],
    out_size=N,
    width_size=10,
    depth=3,
    activation=jax.nn.relu,
    key=k_trunk
)

A = jr.normal(k_T, (N, K))

net_trunk_opt, A_opt, losses1 = train_step_1(net_trunk, A, data, optax.adam(1e-4), 10000)
net_branch_opt, T = train_step_2(net_branch, net_trunk_opt, A_opt, data, optax.adam(1e-4), 10000)
# %%

t_trunk = lambda x: A_opt.T @ net_trunk(x)
def deeponet(input_branch, input_trunk):

    return net_branch_opt(input_branch), net_trunk_opt(input_trunk)

#%% Validation
dataset_test = jnp.load("data/antiderivative_aligned_test.npz", allow_pickle=True)
branch_inputs_test = dataset_test["X"][0]
trunk_inputs_test = dataset_test["X"][1]
outputs_test = dataset_test["y"]

output_branch, output_trunk = jax.vmap(
        deeponet,
        in_axes=(None, 0)
    )(branch_inputs_test[0, :], trunk_inputs_test)

pred = jnp.sum((output_trunk @ A_opt) * output_branch, axis=0)

plt.plot(
    trunk_inputs_test[:, 0],
    branch_inputs_test[0, :],
    label="input",
)
plt.plot(
    trunk_inputs_test[:, 0],
    outputs_test[0, :],
    label="gt antiderivative",
)
plt.plot(
    trunk_inputs_test[:, 0],
    deeponet(branch_inputs_test[0, :], trunk_inputs_test[:, 0]),
    label="prediction",
)
plt.legend()
plt.grid()
# %%
