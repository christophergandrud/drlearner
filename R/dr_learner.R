#' Estimate heterogeneous treatment effect using Doubly Robust Estimation
#' (Kennedy 2022) using `cv.glmnet` for estimate construction
#'
#' @param X matrix of covariates
#' @param Y numeric vector of outcomes
#' @param W numeric vector of treatment states \[0, 1\]. If a logical vector is
#' supplied, will coerce to numeric with `FALSE = 0` and `TRUE = 1`.
#' @param family character in ("gaussian", "binomial") to pass to `cv.glmnet`
#' @param ... arguments to pass to `cv.glmnet`
#'
#' @returns A list of estimates needed for best linear projections of the
#' conditional average treatment effect for approximately optimal targeting
#' The list includes observed outcomes (`Y`), treatments (`W`),
#' estimates of E\[Y | X = x\] (`Y.hat`) and E\[W | X = x\] (`W.hat`),
#' and the localized predictions of the causal forest E\[Y_1 - Y_0 | X = x\]
#' (`tau.hat`)
#'
#' @references Kennedy, Edward H. (2022) "Towards optimal doubly robust
#' estimation of heterogeneous causal effects".
#' <https://arxiv.org/abs/2004.14497>.
#'
#' @seealso cv.glmnet
#'
#' @examples
#' # Simulate data
#' n <- 2000
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' # CATE varies along one dim only.
#' tau_ex <- function(x) {
#'   1 / (1 + exp(-x))
#' }
#' TAU <- tau_ex(X[, 3])
#' # Propensity  and Outcome vary along 2 and 5 dimensions only.
#' W <- rbinom(n, 1, 1 / (1 + exp(-X[, 1] - X[, 2])))
#' Y <- pmax(X[, 2] + X[, 3], 0) + rowMeans(X[, 4:6]) / 2 + W * TAU + rnorm(n)
#'
#' drl <- dr_learner(X, Y, W)
#'
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict
#'
#' @export

dr_learner <- function(X, Y, W, family = "gaussian", ...) {
  # Attempting smart coercion
  if (is.logical(W)) {
    W <- ifelse(W == TRUE, 1, 0)
  }
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }

  # Split into 3 samples
  n <- nrow(X)
  stopifnot(
    "X, Y, and W must all be of the same length" =
      n == length(Y) & n == length(W)
  )
  even_split <- floor(n / 3)
  s <- c(rep(1:3, even_split), 1:(n - even_split * 3))
  s <- sample(s)

  # Step 1
  # Propensity scores
  #### This could be parallelised ####
  pi.hat <- predict(cv.glmnet(X[s == 1, ], W[s == 1], family = "binomial", ...),
    newx = X, type = "response", s = "lambda.min"
  )

  # Y given covariates and treatment assignment
  mu0.hat <- predict(cv.glmnet(X[W == 0 & s == 2, ], Y[W == 0 & s == 2],
    family = family, ...
  ), newx = X, s = "lambda.min")
  mu1.hat <- predict(cv.glmnet(X[W == 1 & s == 2, ], Y[W == 1 & s == 2],
    family = family, ...
  ), newx = X, s = "lambda.min")

  # Step 2
  # Psuedo-regression
  pseudo <- ((W - pi.hat) / (pi.hat * (1 - pi.hat))) * (Y - W * mu1.hat - (1 - W) * mu0.hat) + mu1.hat - mu0.hat
  tau.hat <- predict(cv.glmnet(X[s == 3, ], pseudo[s == 3], family = family, ...), newx = X, s = "lambda.min")

  # Predict Y give X. Needed for best linear projection
  Y.hat <- predict(cv.glmnet(X[s == 3, ], Y[s == 3], family = family, ...),
    newx = X, s = "lambda.min"
  )

  out <- list(
    Y = Y, W = W, Y.hat = Y.hat, W.hat = pi.hat, tau.hat = tau.hat
  )
  class(out) <- "drlearner_blp"
  return(out)
}
