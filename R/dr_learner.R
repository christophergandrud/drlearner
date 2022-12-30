#' Estimate heterogeneous treatment effect using Doubly Robust Estimation
#' (Kennedy 2022)
#'
#' @param X matrix of covariates
#' @param Y numeric vector of outcomes
#' @param W logical vector of treatment states (0, 1)
#' @param family character in ("gaussian", "binomial") to pass to `cv.glmnet`
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
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict
#'
#' @export

df_learner <- function(X, Y, W, family = "gaussian") {
  # Split into 3 samples
  n <- nrow(X)
  stopifnot(
    "X, Y, and W must all be of the same length" =
      n == length(Y) & n == length(W)
  )
  s <- sample(rep(1:3), floor(n / 3))

  # Step 1
  # Propensity scores
  W.hat <- predict(cv.glmnet(X[s == 1, ], W[s == 1], family = "binomial"),
    newx = X, type = "response", s = "lambda.min"
  )

  # Y given covariates and treatment assignment
  mu0.hat <- predict(cv.glmnet(X[W == 0 & s == 2, ], Y[W == 0 & s == 2],
    family = family
  ), newx = X, s = "lambda.min")
  mu1.hat <- predict(cv.glmnet(X[W == 1 & s == 2, ], Y[W == 1 & s == 2],
    family = family
  ), newx = X, s = "lambda.min")

  # Step 2
  # Psuedo-regression
  pseudo <- ((W - W.hat) / (W.hat * (1 - W.hat))) * (Y - W * mu1.hat - (1 - W) * mu0.hat) + mu1.hat - mu0.hat
  tau.hat <- predict(cv.glmnet(X[s == 3, ], pseudo[s == 3]), newx = X, s = "lambda.min")

  # Predict Y give X. Needed for for best linear projection
  Y.hat <- predict(cv.glmnet(X[s == 3, ], Y[s == 3]),
    family = family,
    newx = X, s = "lambda.min"
  )

  out <- list(
    Y.org = Y, W.org = W, Y.hat = Y.hat, W.hat = W.hat,
    tau.hat = tau.hat
  )
  class(out) <- "drlearner_blp"
  return(out)
}