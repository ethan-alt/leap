standir   <- '/proj/ibrahimlab/leap/Stan'
npp.stan  <- cmdstanr::cmdstan_model(file.path(standir, 'npp_prior.stan'))
pbnpp.stan <- cmdstanr::cmdstan_model(file.path(standir, 'pbnpp_prior.stan'))

n0 <- 200
X0 <- cbind(1, rnorm(n0), rnorm(n0))
beta <- rnorm(ncol(X0), mean = 1, sd = 1)
sigma <- 2
y0 <- rnorm(n0, (X0 %*% beta)[, 1], sigma)
standat <- list(
  'n0' = n0
  , 'p' = ncol(X0)
  , 'y0' = y0
  , 'X0' = X0
  , 'prec_shape' = 1
  , 'prec_rate' = 0
  , 'a0_lower' = 0
  , 'a0_upper' = 1
  , 'trtidx'   = 2
)
standat$covidx <- (1:standat$p)[-standat$trtidx]



# fit.npp <- npp.stan$sample(data = standat, iter_warmup = 2000, iter_sampling = 20000, chains = 4, parallel_chains = 4)

fit.hist.pb <- lm(y0 ~ 0 + X0[, standat$covidx])
fit.pbnpp <- pbnpp.stan$sample(data = standat, iter_warmup = 2000, iter_sampling = 20000, chains = 4, parallel_chains = 4)
fit.pbnpp$summary()
coef(fit.hist.pb)
