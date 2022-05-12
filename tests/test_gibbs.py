import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from scipy.linalg import toeplitz

from aemcmc.gibbs import (
    F_matrix_construct,
    R_r,
    bernoulli_horseshoe_gibbs,
    bernoulli_horseshoe_match,
    bernoulli_horseshoe_model,
    dispersion_term_model,
    horseshoe_match,
    horseshoe_model,
    nbinom_horseshoe_gibbs,
    nbinom_horseshoe_gibbs_with_dispersion,
    nbinom_horseshoe_match,
    nbinom_horseshoe_model,
)


@pytest.fixture
def srng():
    return RandomStream(1234)


def test_match_horseshoe(srng):
    horseshoe_match(horseshoe_model(srng))

    # exchanging tau and lmbda in the product
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=size)
    horseshoe_match(beta_rv)


def test_match_horseshoe_wrong_graph(srng):
    beta_rv = srng.normal(0, 1)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_match_horseshoe_wrong_local_scale_dist(srng):
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lmbda_rv = srng.normal(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_match_horseshoe_wrong_global_scale_dist(srng):
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.normal(0, 1, size=1)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_match_horseshoe_wrong_dimensions(srng):
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=size)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)

    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    with pytest.raises(ValueError):
        horseshoe_match(beta_rv)


def test_match_nbinom_horseshoe(srng):
    nbinom_horseshoe_match(nbinom_horseshoe_model(srng))


def test_match_binom_horseshoe_wrong_graph(srng):
    beta = at.vector("beta")
    X = at.matrix("X")
    Y = X @ beta

    with pytest.raises(ValueError):
        nbinom_horseshoe_match(Y)


def test_match_nbinom_horseshoe_wrong_sign(srng):
    X = at.matrix("X")
    h = at.scalar("h")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(2 * eta)
    Y_rv = srng.nbinom(h, p)

    with pytest.raises(ValueError):
        nbinom_horseshoe_match(Y_rv)


def test_horseshoe_nbinom(srng):
    """
    This test example is modified from section 3.2 of Makalic & Schmidt (2016)
    """
    h = 2
    p = 10
    N = 50

    # generate synthetic data
    true_beta = np.array([5, 3, 3, 1, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = srng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = srng.nbinom(h, at.sigmoid(-(X.dot(true_beta))))

    # build the model
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lambda_rv = srng.halfcauchy(0, 1, size=p)
    beta_rv = srng.normal(0, tau_rv * lambda_rv, size=p)

    eta_tt = X @ beta_rv
    p_tt = at.sigmoid(-eta_tt)
    Y_rv = srng.nbinom(h, p_tt)

    # sample from the posterior distributions
    num_samples = at.scalar("num_samples", dtype="int32")
    outputs, updates = nbinom_horseshoe_gibbs(srng, Y_rv, y, num_samples)
    sample_fn = aesara.function((num_samples,), outputs, updates=updates)

    beta, lmbda, tau = sample_fn(2000)

    assert beta.shape == (2000, p)
    assert lmbda.shape == (2000, p)
    assert tau.shape == (2000, 1)

    # test distribution domains
    assert np.all(tau > 0)
    assert np.all(lmbda > 0)


def test_R_r(srng):
    F = F_matrix_construct(50)
    r_ = srng.gamma(1, 1)
    R_r(F, r_, 4)


def test_horseshoe_nbinom_w_dispersion(srng):
    """
    This test example is modified from section 3.2 of Makalic & Schmidt (2016)
    """
    h = 1
    p = 10
    N = 50

    # generate synthetic data
    true_beta = np.array([1, 0, 0, 0, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = srng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = srng.nbinom(h, at.sigmoid(-(X.dot(true_beta))))

    eta_tt = X @ at.as_tensor_variable(true_beta)
    p_tt = at.sigmoid(-eta_tt)
    r_rv = dispersion_term_model(srng)
    Y_rv = srng.nbinom(r_rv, p_tt)

    # sample from the posterior distributions
    num_samples = at.scalar("num_samples", dtype="int32")
    outputs, updates = nbinom_horseshoe_gibbs_with_dispersion(
        srng, Y_rv, y, num_samples
    )

    sample_fn = aesara.function((num_samples,), outputs, updates=updates)

    sample_num = 2000
    r, l_i = sample_fn(sample_num)

    assert r.shape == (sample_num, 1)
    assert l_i.shape == (sample_num, N)

    # test distribution domains
    assert np.all(r > 0)
    assert np.all(l_i >= 0)


def test_match_bernoulli_horseshoe(srng):
    bernoulli_horseshoe_match(bernoulli_horseshoe_model(srng))


def test_match_bernoulli_horseshoe_wrong_graph(srng):
    beta = at.vector("beta")
    X = at.matrix("X")
    Y = X @ beta

    with pytest.raises(ValueError):
        bernoulli_horseshoe_match(Y)


def test_match_bernoulli_horseshoe_wrong_sign(srng):
    X = at.matrix("X")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(2 * eta)
    Y_rv = srng.bernoulli(p)

    with pytest.raises(ValueError):
        bernoulli_horseshoe_match(Y_rv)


def test_bernoulli_horseshoe(srng):
    p = 10
    N = 50

    # generate synthetic data
    true_beta = np.array([5, 3, 3, 1, 1] + [0] * (p - 5))
    S = toeplitz(0.5 ** np.arange(p))
    X = srng.multivariate_normal(np.zeros(p), cov=S, size=N)
    y = srng.bernoulli(at.sigmoid(-X.dot(true_beta)))

    # build the model
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lambda_rv = srng.halfcauchy(0, 1, size=p)
    beta_rv = srng.normal(0, tau_rv * lambda_rv, size=p)

    eta_tt = X @ beta_rv
    p_tt = at.sigmoid(-eta_tt)
    Y_rv = srng.bernoulli(p_tt)

    # sample from the posterior distributions
    num_samples = at.scalar("num_samples", dtype="int32")
    outputs, updates = bernoulli_horseshoe_gibbs(srng, Y_rv, y, num_samples)
    sample_fn = aesara.function((num_samples,), outputs, updates=updates)

    beta, lmbda, tau = sample_fn(2000)

    assert beta.shape == (2000, p)
    assert lmbda.shape == (2000, p)
    assert tau.shape == (2000, 1)

    # test distribution domains
    assert np.all(tau > 0)
    assert np.all(lmbda > 0)
