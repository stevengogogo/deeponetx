import jax
import jax.numpy as jnp
from deeponetx.data import kernels
from jax import config
from jax.experimental import enable_x64
#config.update("jax_enable_x64", True)
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

    # Test operator overloading
    for i in comp_ks:
        for j in comp2_ks: 
            for k in ks: 
                a = (i + j)(1.0, 0.)
                al = jax.vmap(i+j)(x,y)

def test_kernel2d():
    k = kernels.SquaredExponential(0.5, 1.0)
    x = jnp.linspace(0,1,num=100)
    y = jnp.linspace(3,5,num=len(x))
    xv, yv = jnp.meshgrid(x,y)
    X = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
    covm = kernels.cov_matrix(k, X, jitter=1e-13)
    
    assert covm.shape == (len(x)**2, len(x)**2)
    assert jnp.allclose(covm, covm.T)

def test_cov_matrix():
    # Try cholesky
    k = kernels.SquaredExponential(0.5, 1.0)
    x = jnp.linspace(0,1,num=1000)

    with enable_x64(): #temperary for float64
        x_ = x.astype(jnp.float64) #convert to float64
        m_cov = kernels.cov_matrix(k, x_, jitter=1e-13).astype(jnp.float64)
        L = jnp.linalg.cholesky(m_cov, upper=False)
        
        assert jnp.allclose(m_cov, m_cov.T, rtol=1e-05, atol=1e-05)
        #assert jnp.allclose(L, L.T)

    assert m_cov.shape[0] == len(x)
    assert m_cov.shape[1] == len(x)
    assert not jnp.isnan(L).any()