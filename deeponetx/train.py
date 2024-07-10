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
    return jax.vmap(
        jax.vmap(
            model, in_axes=(None, 0)), 
        in_axes=(0, None))(data.input_branch, data.input_trunk)

def loss_fn(model:AbstractDeepONet, data:DataDeepONet):

    preds = predict(model, data)

    mse = jnp.mean(jnp.square(preds - data.output))
    return mse

@eqx.filter_jit
def update_fn(model:AbstractDeepONet, data:DataDeepONet, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, data)
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss


def train(model:AbstractDeepONet, data:DataDeepONet, optimizer, n_iter:int, batch_size:int=None, key=jr.PRNGKey(0)):
    state = optimizer.init(
        eqx.filter(model, eqx.is_array)
        )
    losses = np.zeros(n_iter)
    batch_size = len(data) if batch_size is None else batch_size
    with tqdm(range(n_iter)) as t:
        for i in t:
            k_b, key = jax.random.split(key)
            model, state, loss = update_fn(model, data.sample(), optimizer, state)
            losses[i] = loss
            t.set_description(f'Loss: {loss}\t')
    return model, losses