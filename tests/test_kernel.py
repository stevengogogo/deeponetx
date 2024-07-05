import jax
import jax.numpy as jnp
from deeponetx.data import kernels


def test_kernel():
    ks = [kernels.SquaredExponential(0.5, 1.0),
        kernels.RationalQuadratic(0.5, 1.0, 1.0)
    ]
    comp_ks = [
        kernels.SumKernel(ks[0], ks[1]),
        kernels.ProductKernel(ks[0], ks[1])
    ]
    comp2_ks = [
        kernels.SumKernel(comp_ks[1], comp_ks[0]),
        kernels.ProductKernel(comp_ks[1], comp_ks[0])
    ]

    Ks = ks + comp_ks + comp2_ks
    
    x = jnp.linspace(0,1,num=100)
    y = jnp.linspace(3,5,num=len(x))
    for K in Ks: 
        a = K(1., 0.)
        al = jax.vmap(K)(x,y)

    assert ks[0](x[0], x[0]) == 1.0**2