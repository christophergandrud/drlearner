
<!-- README.md is generated from README.Rmd. Please edit that file -->

# drlearner

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/christophergandrud/dflearner/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/christophergandrud/dflearner/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

Doubly robust machine learner (DR Learner) with sample splitting for
Heterogeneous Treatment Effect Estimation from [Kennedy
(2022)](https://arxiv.org/pdf/2004.14497.pdf) to enable approximately
optimal policy targeting using best linear projections. This
implementation uses
[`cv.glmnet`](https://glmnet.stanford.edu/reference/cv.glmnet.html) for
estimate construction.

## Installation

You can install the development version of drlearner from
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("christophergandrud/drlearner")
```

Because drlearner is intended to work with blpopt for best linear
projection estimation, also install:

``` r
remotes::install_github("christophergandrud/blpopt")
```

## About

Imagine we want to target individuals in a population with a treatment.
The effect of the treatment is heterogeneous, i.e. it is conditional on
the individuals’ characteristics (CATE). However, we don’t fully observe
the characteristics of each individual. How might we make optimal
treatment decisions based on the characteristics we do observe?

This package enables one way (Doubly Robust Learner) to do the first
step in the following process for making approximately optimal treatment
targeting decisions:

1.  Estimate CATE, e.g. using `dr_learner` from this package or
    `causal_forest` from the [grf
    package](https://grf-labs.github.io/grf/).

2.  Find the best linear projection of the CATE given the
    characteristics of the individuals we do observe. You can do this
    using the [blpopt
    package](https://github.com/christophergandrud/blpopt).

#### Why this version of DR Learner with sample splitting and not Causal Forests?

Causal forests are a popular method for estimating CATEs. However, it
can be memory intensive and even intractable using common hardware on
large data sets (e.g. millions of observations). In [various simulation
scenarios](https://github.com/christophergandrud/blpopt/blob/main/notebooks/compare-cate-blp-with-naive-lm.ipynb)
DR Learner produces equivalent or better results (lower mean squared
error) much faster; orders of magnitude faster and for large data sets
(millions of observations) is computationally feasible on a laptop,
while causal forest runs out of memory.

### Formal setup

Imagine data $Z = (X,W,Y)$

$$
\begin{aligned}
\pi(x) & =\mathbb{P}(W=1 \mid X=x) \\
\mu_a(x) & =\mathbb{E}(Y \mid X=x, W=w) \\
\eta(x) & =\mathbb{E}(Y \mid X=x)
\end{aligned}
$$

The aim of conditional average treatment effect estimation is to
estimate the difference of the regression functions under treatment
$(W = 1)$ versus control $(W = 0)$

$$
\tau(x) \equiv \mu_1(x)-\mu_0(x).
$$

From $\tau(x)$ we would like to then identify optimal treatments for
each individual, even when we only observe a subset of covariates $A$
where $A \subseteq X$.

### Doubly Robust Learner (DR Learner) with sample splitting

[Kennedy (2022, 10)](https://arxiv.org/pdf/2004.14497.pdf) proposes
using doubly robust learning with sample slitting to estimate the
conditional average treatment effects. He proposes the following
algorithm:

Let $(D^n_1,D^n_2)$ be two independent samples of $n$ observations of
the data mentioned in the set up.

*Step 1. Nuisance training*

- Construct estimates of $\widehat{\pi}$ of the propensity scores $\pi$
  using $D^n_1$.
- Construct estimates $(\widehat{\mu_{0}},\widehat{\mu_{1}})$ of the
  regression functions $(\mu_0,\mu_1)$ using $D^n_1$.

*Step 2. Pseudo-outcome regression:*

$$
\widehat{\varphi}(Z)=\frac{W-\widehat{\pi}(X)}{\widehat{\pi}(X)\{1-\widehat{\pi}(X)\}}\\{\{Y-\widehat{\mu}_A(X)\\}\}+\widehat{\mu}_1(X)-\widehat{\mu}_0(X)
$$

and regress it on covariates $X$ in the test sample $D^n_2$, giving us

$$
\widehat{\tau}_{d r}(x)=\widehat{\mathbb{E}}_n\\{\widehat{\varphi}(Z) \mid X=x\\}
$$

*Step 3. Estimate outcome given covariates*

Not included in Kennedy (2022), but needed for the best linear
projection: estimate the outcome given the covariates $X$ using sample
$D^n_2$.

*Step 4. Best linear projection of the CATE*

Using the output of the DR Learner procedure above, now find the best
linear projection of the CATE given observed covariates $A$ for unit $i$
$(\widehat{\tau_{BLP}} (A_{i}))$, where $A \subseteq X$.

See
[here](https://github.com/christophergandrud/blpopt/blob/main/notebooks/promises-pitfalls-cate-blp.ipynb)
for a discussion of when this strategy can be problematic.

*Step 5 (optional): Expected benefit of targeting*

It is often non-trivial to apply a treatment targeting regime, even if
we have good estimates of the CATE. We can estimate the expected benefit
of approximately optimal targeting by comparing the the CATE-BLP if we
apply the treatment to the whole population $(Pr(W) = 1)$ compared to a
world where we only apply a treatment if the BLP is greater than some
value $\kappa_i$:

$$
W_i = 
\begin{cases}
& 1, & \text{if}\ \hat{\tau}_{BLP} (A_{i}) > \kappa_i \\
& 0, & \text{otherwise}
\end{cases}.
$$

Typically, $\kappa_i \ge 0$ as we usually don’t want to apply a
treatment with no or negative incremental value. $\kappa_i > 0$ reflects
a non-zero cost of treatment for unit $i$.

We can estimate uncertainty around the total benefit via bootstrapping.
See `blpopt::cate_blp_bootstrap` for an implementation.

## Example

``` r
# remotes::install_github("christophergandrud/blpopt")
# remotes::install_github("christophergandrud/drlearner")

xfun::pkg_attach2("blpopt", "drlearner", "tidyr", "ggplot2")
#> 
#> Attaching package: 'blpopt'
#> The following object is masked from 'package:stats':
#> 
#>     predict
#> The following object is masked from 'package:base':
#> 
#>     summary
theme_set(theme_minimal())

# Simulate data
set.seed(3214)
n <- 2000
p <- 20
X <- matrix(rnorm(n * p), n, p)

# CATE varies along one dim only.
tau_ex <- function(x) {
  (1 / (1 + exp(-x))) - 0.5
}
TAU <- tau_ex(X[, 3])

# Propensity and Outcome vary along only 2 dimensions
W <- rbinom(n, 1, 1 / (1 + exp(-X[, 1] - X[, 2])))
Y <- pmax(X[, 2] + X[, 3], 0) + rowMeans(X[, 4:6]) / 2 + W * TAU + rnorm(n)

# CATE-BLP with DR Learner (Kennedy 2022)
drl <- dr_learner(X, Y, W)
blp_drl <- cate_blp(drl, X[, 3])
blp_drl_pred <- tibble(blp_drl = predict(blp_drl)$predicted, A = X[, 3])

ggplot(blp_drl_pred, aes(A, blp_drl)) +
  geom_function(fun = tau_ex, col = "black", linetype = "dashed") +
  geom_line(color = "green") +
  ylab("Predicted Treatment Effect\n") +
  xlab("\nObserved Covariate Values") +
  ggtitle("[1 Simulation] Compare predicted treatment effects from\nCATE-BLP and plugin linear regression ",
    subtitle = "Black dashed line represents the true effect.\nGreen line is CATE-BLP from DR Learner"
  )
```

<img src="man/figures/README-unnamed-chunk-2-1.png" width="100%" />
