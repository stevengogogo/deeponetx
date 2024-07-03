import jax
import jax.numpy as jnp
import optax 
import equinox as eqx
from typing import NamedTuple
from tqdm.auto import tqdm
import numpy as np
from .nn import AbstractDeepONet

class DataDeepONet(NamedTuple):
    """Data for DeepONet

    Args:
        - `input_branch`: jnp.ndarray # conditioned function (sampled) [nsample, ngrid]
        - `input_trunk`: jnp.ndarray # location y [nsample, locations]
        - `output`: jnp.array # output of operator at location y [nsample, locations]
    """
    input_branch: jnp.ndarray
    input_trunk: jnp.ndarray
    output: jnp.array
    def __getitem__(self, key):
        """Get subset of data
        """
        return DataDeepONet(self.input_branch[key], self.input_trunk, self.output[key])

def loss_fn(model:AbstractDeepONet, data:DataDeepONet):

    preds = jax.vmap( 
        jax.vmap(
            model, in_axes=(None, 0)), 
        in_axes=(0, None))(data.input_branch, data.input_trunk)

    mse = jnp.mean(jnp.square(preds - data.output))
    return mse

@eqx.filter_jit
def update_fn(model:AbstractDeepONet, data, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, data)
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss


def train(model:AbstractDeepONet, data:DataDeepONet, optimizer, n_iter:int):
    state = optimizer.init(
        eqx.filter(model, eqx.is_array)
        )
    losses = np.zeros(n_iter)
    with tqdm(range(n_iter)) as t:
        for i in tqdm(range(n_iter)):
            model, state, loss = update_fn(model, data, optimizer, state)
            losses[i] = loss
            t.set_description(f'Loss: {loss}')
    return model, losses