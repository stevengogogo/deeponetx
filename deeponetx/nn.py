import jax
import jax.numpy as jnp
import equinox as eqx 
import abc 

class AbstractDeepONet(eqx.Module):
    net_branch: eqx.Module
    net_trunk: eqx.Module

    @abc.abstractmethod
    def __init__():
        raise NotImplementedError 
    
    @abc.abstractmethod
    def __call__():
        raise NotImplementedError
    

class UnstackDeepONet(AbstractDeepONet):
    net_branch: eqx.Module
    net_trunk: eqx.Module
    bias: jax.Array

    def __init__(self, net_branch, net_trunk, bias):
        self.net_branch = net_branch
        self.net_trunk = net_trunk
        self.bias = bias

    def __call__(self, x_branch, x_trunk):
        out_branch = self.net_branch(x_branch)
        out_trunk = self.net_trunk(x_trunk)
        inner_prod = jnp.sum(out_branch * out_trunk, keepdims=True)
        return (inner_prod + self.bias)[0]
    

def create_UnstackDeepONet1d_MLP(in_size_branch, width_size, depth, interact_size, activation, *, key):
    """Creat deeponet for 1D problem. Two MLP used for branch and trunk are constructed similarly
    """
    key_b, key_t, key_bs = jax.random.split(key, num=3)
    net_branch = eqx.nn.MLP(
        in_size=in_size_branch,
        out_size=interact_size,
        width_size=width_size,
        depth=depth,
        activation=activation,
        key=key_b
    )
    net_trunk = eqx.nn.MLP(
        in_size=1,
        out_size=interact_size,
        width_size=width_size,
        depth=depth,
        activation=activation,
        final_activation=activation,
        key=key_t
    )
    bias = jax.random.uniform(key_bs, shape=(1,))
    return UnstackDeepONet(net_branch, net_trunk, bias)
    