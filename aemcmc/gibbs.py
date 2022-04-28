from typing import Dict, List, Tuple, Union

import aesara
import aesara.tensor as at
import numpy as np
from aesara.graph import optimize_graph
from aesara.graph.unify import eval_if_etuple
from aesara.ifelse import ifelse
from aesara.tensor.math import Dot
from aesara.tensor.random import RandomStream
from aesara.tensor.var import TensorVariable
from etuples import etuple, etuplize
from unification import unify, var

from aemcmc.dists import (
    multichoice,
    multivariate_normal_cong2017,
    multivariate_normal_rue2005,
    polyagamma,
)


def update_beta_low_dimension(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    Q = X.T @ (omega[:, None] * X)
    indices = at.arange(Q.shape[1])
    Q = at.subtensor.set_subtensor(
        Q[indices, indices],
        at.diag(Q) + lmbdatau_inv,
    )
    return multivariate_normal_rue2005(srng, X.T @ (omega * z), Q)


def update_beta_high_dimension(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    return multivariate_normal_cong2017(srng, lmbdatau_inv, omega, X, z)


def update_beta(
    srng: RandomStream,
    omega: TensorVariable,
    lmbdatau_inv: TensorVariable,
    X: TensorVariable,
    z: TensorVariable,
) -> TensorVariable:
    return ifelse(
        X.shape[1] > X.shape[0],
        update_beta_high_dimension(srng, omega, lmbdatau_inv, X, z),
        update_beta_low_dimension(srng, omega, lmbdatau_inv, X, z),
    )


halfcauchy_1_lv, halfcauchy_2_lv = var(), var()
zero_lv = var()
horseshoe_pattern = etuple(
    etuplize(at.random.normal),
    var(),
    var(),
    var(),
    zero_lv,
    etuple(etuplize(at.mul), halfcauchy_1_lv, halfcauchy_2_lv),
)


def horseshoe_model(srng: TensorVariable) -> TensorVariable:
    """Horseshoe shrinkage prior [1]_.

    References
    ----------
    .. [1]: Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010).
            The horseshoe estimator for sparse signals.
            Biometrika, 97(2), 465-480.

    """
    size = at.scalar("size", dtype="int32")
    tau_rv = srng.halfcauchy(0, 1, size=1)
    lmbda_rv = srng.halfcauchy(0, 1, size=size)
    beta_rv = srng.normal(0, tau_rv * lmbda_rv, size=size)
    return beta_rv


def horseshoe_match(graph: TensorVariable) -> Tuple[TensorVariable, TensorVariable]:
    graph_opt = optimize_graph(graph)
    graph_et = etuplize(graph_opt)

    s = unify(graph_et, horseshoe_pattern)
    if s is False:
        raise ValueError("Not a horseshoe prior.")

    halfcauchy_1 = eval_if_etuple(s[halfcauchy_1_lv])
    if halfcauchy_1.owner is None or not isinstance(
        halfcauchy_1.owner.op, type(at.random.halfcauchy)
    ):
        raise ValueError(
            "Not a horseshoe prior. One of the shrinkage parameters "
            + "in your model is not half-Cauchy distributed."
        )

    halfcauchy_2 = eval_if_etuple(s[halfcauchy_2_lv])

    if halfcauchy_2.owner is None or not isinstance(
        halfcauchy_2.owner.op, type(at.random.halfcauchy)
    ):
        raise ValueError(
            "Not a horseshoe prior. One of the shrinkage parameters "
            + "in your model is not half-Cauchy distributed."
        )

    if halfcauchy_1.type.shape == (1,):
        lmbda_rv = halfcauchy_2
        tau_rv = halfcauchy_1
    elif halfcauchy_2.type.shape == (1,):
        lmbda_rv = halfcauchy_1
        tau_rv = halfcauchy_2
    else:
        raise ValueError(
            "Not a horseshoe prior. The global shrinkage parameter "
            + "in your model must be one-dimensional."
        )

    return (lmbda_rv, tau_rv)


def horseshoe_step(
    srng: RandomStream,
    beta: TensorVariable,
    sigma: TensorVariable,
    lmbda_inv: TensorVariable,
    tau_inv: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    r"""Gibbs kernel to sample from the posterior distribution of a horseshoe prior.

    This kernel generates samples from the posterior distribution of the local
    and global shrinkage parameters of a horseshoe prior, respectively :math:`\lambda`
    and :math:`\tau` in the following model:

    .. math::

        \begin{align*}
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2\;\sigma^2)\\
            \sigma^2 &\sim \sigma^{-2} \mathrm{d} \sigma\\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}

    We use the following observations [1]_ to sample from the posterior
    conditional probability of :math:`\tau` and :math:`\lambda`:

    1. The half-Cauchy distribution can be intepreted as a mixture of inverse-gamma
    distributions;
    2. If :math:` Y \sim InverseGamma(1, a)`, :math:`Y \sim 1 / \operatorname{Exp}(a)`.

    Parameters
    ----------
    srng
        The random number generating object to be used during sampling.
    beta
        Regression coefficients.
    sigma2
        Variance of the regression coefficients.
    lmbda2_inv
        Square inverse of the local shrinkage parameters.
    tau2_inv
        Square inverse of the global shrinkage parameters.

    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

    """
    upsilon_inv = srng.exponential(1 + lmbda_inv)
    zeta_inv = srng.exponential(1 + tau_inv)

    beta2 = beta * beta
    lmbda_inv_new = srng.exponential(upsilon_inv + 0.5 * beta2 * tau_inv / sigma)
    tau_inv_new = srng.gamma(
        0.5 * (beta.shape[0] + 1),
        zeta_inv + 0.5 * (beta2 * lmbda_inv_new).sum() / sigma,
    )
    return lmbda_inv_new, tau_inv_new


X_lv = var()
beta_lv = var()
neg_one_lv = var()

sigmoid_dot_pattern = etuple(
    etuplize(at.sigmoid),
    etuple(etuplize(at.mul), neg_one_lv, etuple(etuple(Dot), X_lv, beta_lv)),
)


h_lv = var()
nbinom_sigmoid_dot_pattern = etuple(
    etuplize(at.random.nbinom), var(), var(), var(), h_lv, sigmoid_dot_pattern
)


def nbinom_sigmoid_dot_match(
    graph: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
    graph_opt = optimize_graph(graph)
    graph_et = etuplize(graph_opt)
    s = unify(graph_et, nbinom_sigmoid_dot_pattern)
    if s is False:
        raise ValueError("Not a negative binomial regression.")

    if all(s[neg_one_lv].data != -1):
        raise ValueError(
            "Not a negative binomial regression. The argument to "
            + "the sigmoid must be minus the dot product."
        )

    h = eval_if_etuple(s[h_lv])
    beta_rv = eval_if_etuple(s[beta_lv])
    X = eval_if_etuple(s[X_lv])

    return X, h, beta_rv


def dispersion_term_model(srng):
    h_rv = srng.gamma(1, 1, size=1)  # a_0 = 1
    r_rv = srng.gamma(1, 1 / h_rv, size=1)  # b_0 = 1
    return r_rv


def F_matrix_construct(N):
    """
    function to create F matrix for sampling step of L_i

    F matrix is a lower triangular matrix with F(1, 1) = 1, and F(m, j) =0 if j > m

    Parameters
    ----------
    N
        size of the matrix, defined by the maximum of the observation
    """
    F = np.zeros((N, N))
    F[0, 0] = 1
    for i in range(1, N):
        for j in range(i + 1):
            F[i, j] = i / (i + 1) * F[i - 1, j] + 1 / (i + 1) * F[i - 1, j - 1]
    return F


def R_r(F, r_, row):
    n = F.shape[0]
    if row == 0:
        return F[0, :]
    F = at.as_tensor_variable(F)
    r_factor = at.exp(at.log(r_) * at.arange(1, n + 1))
    r_factor = at.switch(at.lt(r_factor, 1), r_factor, 1)
    dinominator = F[row - 1, :] @ r_factor
    nominator = F[row - 1, :] * r_factor
    return nominator / dinominator


def sampling_for_li(srng, F, y, r):
    N = F.shape[0]
    li = srng.gen(multichoice, N, 1, True, R_r(F, r, y)).T[0].squeeze()
    return li


def dispersion_term_step(
    srng: RandomStream,
    r: TensorVariable,
    l: TensorVariable,
    p: TensorVariable,
    F: TensorVariable,
    y: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    r"""Gibbs kernel to sample from the posterior distribution of a dispersion term r.

    This kernel generates samples from the posterior distribution of the dispersion term
    of negative binomial distribution ie, NB(r, p)

    The approach here used the Compound Poisson representation for the sampling steps.

    we used the following sampling steps for the posterior [1]

    1. `r|  \sim Gamma(a_o + \Sum_{i=1}^N L_i, \frac{1}{h - \sum_{i=1}^N ln(1-p)})`
    2. `Pr(L_i = j |-) == R_r(y_i, j )`, where ` R_r(m, j) = F(m , j) r^j/\sum_{j' = 1}^m(m, j') r^j`,
     where `F(m, j)` a specific kind of matrix defined in the paper.
     see `F_matrix_construct` for detail.
    3. `h \sim Gamma(a_0+ b_0, 1/(g_0 + r))

    Parameters
    ----------
    srng
        The random number generating object to be used during sampling.
    r
        the dispersion parameter
    l
        The number terms of L/N of a Poisson distributed variable in Compound Poisson distribution
    p
        The probability term of i.i.d Logarithmic distributions of Compound Poisson distribution,
        which is also the probability term in N(r, p)
    F
        F matrix for L_i sampling step, constant.
    y
        observations

    References
    ----------
    ..[1] Zhou M, Li L, Dunson D, Carin L. Lognormal and Gamma Mixed Negative Binomial Regression.

    """
    h = srng.gamma(1e-2 + 1 * 1e-2, 1 / (1e-2 + r))
    r = srng.gamma(1e-2 + l.sum(), 1 / (h - at.log(1 - p).sum()))
    l = sampling_for_li(srng, F, y, r).astype(aesara.config.floatX)
    return r, l


def nbinom_horseshoe_model_with_dispersion(srng: RandomStream) -> TensorVariable:
    """Negative binomial regression model with a horseshoe shrinkage prior."""
    X = at.matrix("X")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    r = dispersion_term_model(srng)
    Y_rv = srng.nbinom(r, p)
    return Y_rv


def nbinom_horseshoe_model(srng: RandomStream) -> TensorVariable:
    """Negative binomial regression model with a horseshoe shrinkage prior."""
    X = at.matrix("X")
    h = at.scalar("h")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.nbinom(h, p)

    return Y_rv


def nbinom_horseshoe_match(
    Y_rv: TensorVariable,
) -> Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
]:
    X, h, beta_rv = nbinom_sigmoid_dot_match(Y_rv)
    lmbda_rv, tau_rv = horseshoe_match(beta_rv)
    return h, X, beta_rv, lmbda_rv, tau_rv


def nbinom_horseshoe_match_with_dispersion(
    Y_rv: TensorVariable,
) -> Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
]:
    X, r_rv, beta_rv = nbinom_sigmoid_dot_match(Y_rv)
    lmbda_rv, tau_rv = horseshoe_match(beta_rv)
    return X, beta_rv, lmbda_rv, tau_rv, r_rv


def nbinom_horseshoe_gibbs(
    srng: RandomStream, Y_rv: TensorVariable, y: TensorVariable, num_samples: int
) -> Tuple[Union[TensorVariable, List[TensorVariable]], Dict]:
    r"""Build a Gibbs sampler for the negative binomial regression with a horseshoe prior.

    The implementation follows the sampler described in [1]. It is designed to
    sample efficiently from the following negative binomial regression model:

    .. math::

        \begin{align*}
            y_i &\sim \operatorname{NegativeBinomial}\left(\pi_i, h\right)\\
            h &\sim \pi_h(h) \mathrm{d}h\\
            \pi_i &= \frac{\exp(\psi_i)}{1 + \exp(\psi_i)}\\
            \psi_i &= x^T \beta\\
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2)\\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}


    Parameters
    ----------
    srng: symbolic random number generator
        The random number generating object to be used during sampling.
    Y_rv
        Model graph.
    y: TensorVariable
        The observed count data.
    n_samples: TensorVariable
        A tensor describing the number of posterior samples to generate.

    Returns
    -------
    (outputs, updates): tuple
        A symbolic description of the sampling result to be used to
        compile a sampling function.

    Notes
    -----
    The ``z`` expression in section 2.2 of [1] seems to
    omit division by the Polya-Gamma auxilliary variables whereas [2] and [3]
    explicitely include it. We found that including the division results in
    accurate posterior samples for the regression coefficients. It is also
    worth noting that the :math:`\sigma^2` parameter is not sampled directly
    in the negative binomial regression problem and thus set to 1 [2].

    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    ..[2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.
    ..[3] Neelon, Brian. (2019). Bayesian Zero-Inflated Negative Binomial
          Regression Based on Pólya-Gamma Mixtures. Bayesian Anal.
          2019 September ; 14(3): 829–855. doi:10.1214/18-ba1132.

    """

    def nbinom_horseshoe_step(
        beta: TensorVariable,
        lmbda: TensorVariable,
        tau: TensorVariable,
        y: TensorVariable,
        X: TensorVariable,
        h: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        """Complete one full update of the gibbs sampler and return the new state
        of the posterior conditional parameters.

        Parameters
        ----------
        beta: Tensorvariable
            Coefficients (other than intercept) of the regression model.
        lmbda
            Inverse of the local shrinkage parameter of the horseshoe prior.
        tau
            Inverse of the global shrinkage parameters of the horseshoe prior.
        y: TensorVariable
            The observed count data.
        X: TensorVariable
            The covariate matrix.
        h: TensorVariable
            The "number of successes" parameter of the negative binomial disribution
            used to model the data.

        """
        xb = X @ beta
        w = srng.gen(polyagamma, y + h, xb)
        z = 0.5 * (y - h) / w

        lmbda_inv = 1.0 / lmbda
        tau_inv = 1.0 / tau
        beta_new = update_beta(srng, w, lmbda_inv * tau_inv, X, z)

        lmbda_inv_new, tau_inv_new = horseshoe_step(
            srng, beta_new, 1.0, lmbda_inv, tau_inv
        )
        return beta_new, 1.0 / lmbda_inv_new, 1.0 / tau_inv_new

    h, X, beta_rv, lmbda_rv, tau_rv = nbinom_horseshoe_match(Y_rv)

    outputs, updates = aesara.scan(
        nbinom_horseshoe_step,
        outputs_info=[beta_rv, lmbda_rv, tau_rv],
        non_sequences=[y, X, h],
        n_steps=num_samples,
        strict=True,
    )

    return outputs, updates


def nbinom_horseshoe_gibbs_with_dispersion(
    srng: RandomStream, Y_rv: TensorVariable, y: TensorVariable, num_samples: int
) -> Tuple[Union[TensorVariable, List[TensorVariable]], Dict]:
    r"""Build a Gibbs sampler for the negative binomial regression with a horseshoe prior.

    The implementation follows the sampler described in [1]. It is designed to
    sample efficiently from the following negative binomial regression model:

    .. math::

        \begin{align*}
            y_i &\sim \operatorname{NegativeBinomial}\left(\pi_i, h\right)\\
            h &\sim \pi_h(h) \mathrm{d}h\\
            \pi_i &= \frac{\exp(\psi_i)}{1 + \exp(\psi_i)}\\
            \psi_i &= x^T \beta\\
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2)\\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}


    Parameters
    ----------
    srng: symbolic random number generator
        The random number generating object to be used during sampling.
    Y_rv
        Model graph.
    y: TensorVariable
        The observed count data.
    n_samples: TensorVariable
        A tensor describing the number of posterior samples to generate.

    Returns
    -------
    (outputs, updates): tuple
        A symbolic description of the sampling result to be used to
        compile a sampling function.

    Notes
    -----
    The ``z`` expression in section 2.2 of [1] seems to
    omit division by the Polya-Gamma auxilliary variables whereas [2] and [3]
    explicitely include it. We found that including the division results in
    accurate posterior samples for the regression coefficients. It is also
    worth noting that the :math:`\sigma^2` parameter is not sampled directly
    in the negative binomial regression problem and thus set to 1 [2].

    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    ..[2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.
    ..[3] Neelon, Brian. (2019). Bayesian Zero-Inflated Negative Binomial
          Regression Based on Pólya-Gamma Mixtures. Bayesian Anal.
          2019 September ; 14(3): 829–855. doi:10.1214/18-ba1132.

    """
    N = y.max().eval()
    F = F_matrix_construct(N)

    def nbinom_horseshoe_step(
        beta: TensorVariable,
        lmbda: TensorVariable,
        tau: TensorVariable,
        r: TensorVariable,
        l: TensorVariable,
        y: TensorVariable,
        X: TensorVariable,
    ) -> Tuple[
        TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
    ]:
        """Complete one full update of the gibbs sampler and return the new state
        of the posterior conditional parameters.

        Parameters
        ----------
        beta: Tensorvariable
            Coefficients (other than intercept) of the regression model.
        lmbda
            Inverse of the local shrinkage parameter of the horseshoe prior.
        tau
            Inverse of the global shrinkage parameters of the horseshoe prior.
        y: TensorVariable
            The observed count data.
        X: TensorVariable
            The covariate matrix.
        r: TensorVariable
            The "number of successes" parameter of the negative binomial disribution
            used to model the data.

        """
        xb = X @ beta
        w = srng.gen(polyagamma, y + r, xb)
        z = 0.5 * (y - r) / w

        lmbda_inv = 1.0 / lmbda
        tau_inv = 1.0 / tau
        beta_new = update_beta(srng, w, lmbda_inv * tau_inv, X, z)

        lmbda_inv_new, tau_inv_new = horseshoe_step(
            srng, beta_new, 1.0, lmbda_inv, tau_inv
        )
        eta = X @ beta_new
        p = at.sigmoid(-eta)

        r_new, l_new = dispersion_term_step(srng, r, l, p, F, y)

        return beta_new, 1.0 / lmbda_inv_new, 1.0 / tau_inv_new, r_new, l_new

    X, beta_rv, lmbda_rv, tau_rv, r_rv = nbinom_horseshoe_match_with_dispersion(Y_rv)

    # init l_rv from p
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    l_rv = srng.poisson(-r_rv * at.log(1 - p), size=N).astype(aesara.config.floatX)

    outputs, updates = aesara.scan(
        nbinom_horseshoe_step,
        outputs_info=[beta_rv, lmbda_rv, tau_rv, r_rv, l_rv],
        non_sequences=[y, X],
        n_steps=num_samples,
        strict=True,
    )

    return outputs, updates


bernoulli_sigmoid_dot_pattern = etuple(
    etuplize(at.random.bernoulli), var(), var(), var(), sigmoid_dot_pattern
)


def bernoulli_sigmoid_dot_match(
    graph: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    graph_opt = optimize_graph(graph)
    graph_et = etuplize(graph_opt)
    s = unify(graph_et, bernoulli_sigmoid_dot_pattern)
    if s is False:
        raise ValueError("Not a Bernoulli regression.")

    if all(s[neg_one_lv].data != -1):
        raise ValueError(
            "Not a Bernoulli regression. The argument to the sigmoid "
            + "must be minus the dot product."
        )

    beta_rv = eval_if_etuple(s[beta_lv])
    X = eval_if_etuple(s[X_lv])

    return X, beta_rv


def bernoulli_horseshoe_model(srng: RandomStream) -> TensorVariable:
    """Bernoulli regression model with a horseshoe shrinkage prior."""
    X = at.matrix("X")

    beta_rv = horseshoe_model(srng)
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.bernoulli(p)

    return Y_rv


def bernoulli_horseshoe_match(
    Y_rv: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
    X, beta_rv = bernoulli_sigmoid_dot_match(Y_rv)
    lmbda_rv, tau_rv = horseshoe_match(beta_rv)

    return X, beta_rv, lmbda_rv, tau_rv


def bernoulli_horseshoe_gibbs(
    srng: RandomStream, Y_rv: TensorVariable, y: TensorVariable, num_samples: int
) -> Tuple[Union[TensorVariable, List[TensorVariable]], Dict]:
    r"""Build a Gibbs sampler for bernoulli (logistic) regression with a horseshoe prior.

    The implementation follows the sampler described in [1]. It is designed to
    sample efficiently from the following binary logistic regression model:

    .. math::

        \begin{align*}
            y_i &\sim \operatorname{Bernoulli}\left(\pi_i\right)\\
            \pi &= \frac{1}{1 + \exp\left(-(\beta_0 + x_i^T\,\beta)\right)}\\
            \beta_j &\sim \operatorname{Normal}(0, \lambda_j^2\;\tau^2)\\
            \lambda_j &\sim \operatorname{HalfCauchy}(0, 1)\\
            \tau &\sim \operatorname{HalfCauchy}(0, 1)
        \end{align*}

    Parameters
    ----------
    srng: symbolic random number generator
        The random number generating object to be used during sampling.
    Y_rv
        Model graph.
    y: TensorVariable
        The observed binary data.
    X: TensorVariable
        The covariate matrix.
    n_samples: TensorVariable
        A tensor describing the number of posterior samples to generate.

    Returns
    -------
    (outputs, updates): tuple
        A symbolic description of the sampling result to be used to
        compile a sampling function.


    References
    ----------
    ..[1] Makalic, Enes & Schmidt, Daniel. (2015). A Simple Sampler for the
          Horseshoe Estimator. 10.1109/LSP.2015.2503725.
    ..[2] Makalic, Enes & Schmidt, Daniel. (2016). High-Dimensional Bayesian
          Regularised Regression with the BayesReg Package.

    """

    def bernoulli_horseshoe_step(
        beta: TensorVariable,
        lmbda: TensorVariable,
        tau: TensorVariable,
        y: TensorVariable,
        X: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        """Complete one full update of the gibbs sampler and return the new
        state of the posterior conditional parameters.

        Parameters
        ----------
        beta
            Coefficients (other than intercept) of the regression model.
        lmbda
            Square of the local shrinkage parameter of the horseshoe prior.
        tau
            Square of the global shrinkage parameters of the horseshoe prior.
        y: TensorVariable
            The observed binary data
        X: TensorVariable
            The covariate matrix.

        """
        xb = X @ beta
        w = srng.gen(polyagamma, 1, xb)
        z = 0.5 * y / w

        lmbda_inv = 1.0 / lmbda
        tau_inv = 1.0 / tau
        beta_new = update_beta(srng, w, lmbda_inv * tau_inv, X, z)

        lmbda_inv_new, tau_inv_new = horseshoe_step(
            srng, beta_new, 1.0, lmbda_inv, tau_inv
        )

        return beta_new, 1 / lmbda_inv_new, 1.0 / tau_inv_new

    X, beta_rv, lmbda_rv, tau_rv = bernoulli_horseshoe_match(Y_rv)

    outputs, updates = aesara.scan(
        bernoulli_horseshoe_step,
        outputs_info=[beta_rv, lmbda_rv, tau_rv],
        non_sequences=[y, X],
        n_steps=num_samples,
        strict=True,
    )

    return outputs, updates
