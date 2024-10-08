import jax
import jax.numpy as jnp
import jax.random as jr
import optax 
import equinox as eqx
from tqdm.auto import tqdm
import numpy as np
from .nn import AbstractDeepONet
from .data.data import DatasetDeepONet, DataDeepONet


def predict(model:AbstractDeepONet, data:DatasetDeepONet):
    return jax.vmap(model, in_axes=(0,0))(data.input_branch, data.input_trunk)

def loss_fn(model:AbstractDeepONet, data:DataDeepONet):

    preds = predict(model, data)

    mse = jnp.mean(jnp.square(preds.flatten() - data.output.flatten()))
    return mse

@eqx.filter_jit
def update_fn(model:AbstractDeepONet, data:DataDeepONet, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, data)
    #jax.debug.print("{w}", w=grad.net_branch.layers[0].weight[0])
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss


def train(model:AbstractDeepONet, dataloader:DatasetDeepONet, optimizer, n_iter:int):
    data_iter = iter(dataloader)
    state = optimizer.init(
        eqx.filter(model, eqx.is_array)
        )
    losses = np.zeros(n_iter)
    with tqdm(range(n_iter)) as t:
        for i in t:
            batch = next(data_iter)
            model, state, loss = update_fn(model, batch, optimizer, state)
            losses[i] = loss
            t.set_description(f'Loss: {loss}\t')
    return model, losses