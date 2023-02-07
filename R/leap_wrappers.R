library(psrwe)

## get stan data files
library(cmdstanr)
library(tidyverse)
library(overlapping)
library(lattice)
library(posterior)
library(bayesplot)
library(formula.tools)
standir    <- '/proj/ibrahimlab/leap/Stan'
leap       <- cmdstanr::cmdstan_model(file.path(standir, 'leap.stan'))
pbnpp      <- cmdstanr::cmdstan_model(file.path(standir, 'pbnpp.stan'))
refprior   <- cmdstanr::cmdstan_model(file.path(standir, 'refprior.stan'))
psipp.stan <- cmdstanr::cmdstan_model(file.path(standir, 'psipp.stan'))
leap2      <- cmdstanr::cmdstan_model(file.path(standir, 'leap2_ctrlsOnly.stan'))
samplelist <- list(
  leap = leap, pbnpp = pbnpp, refprior = refprior, leap2 = leap2
)


#' Internal function to obtain standata for regression models
#' 
#' @param formula a formula giving how the outcome is related to covariates and treatment
#' @param data current data set
#' @param histdata historical data
#' @param beta.mean mean for normal prior on regression coefficients
#' @param beta.sd standard deviation for normal prior on regression coefficients
#' @param sigma.mean mean hyperparameter for half-normal prior on standard deviation
#' @param sigma.sd standard deviation hyperparameter for half-normal prior on standard deviation
#' @param prob.conc concentration parameter for Dirichlet prior. If length == 2, a beta prior is used with `shape1 = prob.conc[1]` and `shape2 = prob.conc[2]`. This parameter determines the number of components for the LEAP
get.standata <- function(
  formula, data, histdata
  , beta.mean, beta.sd
  , sigma.mean, sigma.sd
  , prec.shape, prec.rate
  , prob.conc
  , gamma.lower, gamma.upper
  , a0.shape1, a0.shape2
  , a0.lower, a0.upper
  , trtname
  , compute.loglik
  , compute.ssc
) {
  yname <- all.vars(formula)[1]
  X  <- model.matrix(formula, data)
  y  <- data[[yname]]
  y0 <- NULL
  X0 <- NULL
  n0 <- NULL
  if (!(is.null(histdata))) {
    X0 <- model.matrix(formula, histdata)
    y0 <- histdata[[yname]]
    n0 <- length(y0)
  }
  trtidx <- 1
  covidx <- 1:ncol(X)
  formula.terms <- terms(formula)
  if (!(is.null(trtname))) {
    trtidx <- attr(formula.terms, 'intercept') + which(attr(formula.terms, 'term.labels') == trtname)
    covidx <- seq_len(ncol(X))[-trtidx]
  }
  list(
      'n'  = nrow(X)
    , 'n0' = n0
    , 'p'  = ncol(X)
    , 'y'  = y
    , 'y0' = y0
    , 'X'  = X
    , 'X0' = X0
    , 'beta_mean'    = beta.mean
    , 'beta_sd'      = beta.sd
    , 'sigma_mean'   = sigma.mean
    , 'sigma_sd'     = sigma.sd
    , 'prec_shape'   = prec.shape
    , 'prec_rate'    = prec.rate
    , 'conc'         = prob.conc
    , 'K'            = length(prob.conc)
    , 'gamma_lower'  = gamma.lower
    , 'gamma_upper'  = gamma.upper
    , 'a0_shape1'    = a0.shape1
    , 'a0_shape2'    = a0.shape2
    , 'a0_lower'     = a0.lower
    , 'a0_upper'     = a0.upper
    , 'trtidx'       = trtidx
    , 'covidx'       = covidx
    , 'compute_loglik' = as.numeric(compute.loglik)
    , 'compute_ssc' = as.numeric(compute.ssc)
  )
}









#' Obtain posterior samples
#' 
#' Samples from the posterior distribution of a normal linear model for the
#' LEAP, normalized conditional power prior, and a normal prior that does not use
#' historical data
#' 
#' @param formula a formula giving the response and covariates
#' @param data a data frame for the current data set
#' @param histdata a data frame for the historical data set. Ignored if `prior == "refprior"`
#' @param prior the prior to use. Must be one of `"leap"`, `"pbnpp"`, and `"refprior"` for the LEAP, normalized conditional power prior, and reference prior, respectively
#' @param trtname name of treatment indicator. Required if `prior == pbnpp` and otherwise ignored.
#' @param beta.mean scalar giving prior mean for regression coefficients. Ignored if `prior == "pbnpp"`
#' @param beta.sd scalar giving prior standard deviation for regression coefficients. Ignored if `prior == "pbnpp"`
#' @param sigma.mean mean hyperparameter for half-normal prior on standard deviation
#' @param sigma.sd standard deviation hyperparameter for half-normal prior on standard deviation
#' @param prob.conc for the LEAP, the concentration parameter for Dirichlet prior. If length == 2, a beta prior is used with `shape1 = prob.conc[1]` and `shape2 = prob.conc[2]`. This parameter determines the number of components for the LEAP. By default, 2 components are used and a uniform prior is elicited.
#' @param gamma.lower for the LEAP, the lower bound for the beta prior on exchangeability parameter (gamma). Ignored if `prior != "leap"`. By default, no truncation is performed on the prior.
#' @param gamma.lower for the LEAP, the upper bound for the beta prior on exchangeability parameter (gamma). Ignored if `prior != "leap"`. By default, no truncation is performed on the prior.
#' @param a0.shape1 for the LEAP, first shape parameter for the beta prior on the power prior parameter (a0). Ignored if `prior != "pbnpp"`
#' @param a0.shape2 for the LEAP, second shape parameter for the beta prior on the power prior parameter (a0). Ignored if `prior != "pbnpp"`
#' @param a0.lower for the LEAP, the lower bound for the beta prior on the power prior parameter (a0). Ignored if `prior != "pbnpp"`. By default, no truncation is performed on the prior.
#' @param a0.lower for the LEAP, the upper bound for the beta prior on the power prior parameter (a0). Ignored if `prior != "pbnpp"`. By default, no truncation is performed on the prior.
#' @param compute.loglik whether to compute the log likelihood for the current data. This is only useful for leave-one-out (loo)
#' @param compute.ssc whether to compute sample size contribution (SSC). If `prior == 'leap'`, additionally classifies all observations in the historical data.
#' @param ... other objects to pass onto `cmdstanr::sampling`
#' 
#' @return object of type `cmdstanr`
get.samples <- function(
  formula, data, histdata = NULL
  , prior
  , trtname = NULL
  , beta.mean = 0, beta.sd = 10
  , sigma.mean = 0, sigma.sd = 10
  , prec.shape = 0.1, prec.rate = 0.1
  , prob.conc = c(1, 1)
  , gamma.lower = 0, gamma.upper = 1
  , a0.shape1 = 1, a0.shape2 = 1
  , a0.lower = 0, a0.upper = 1
  , compute.loglik = FALSE, compute.ssc = FALSE
  , ...
) {
  standat <- get.standata(
    formula, data, histdata
    , beta.mean, beta.sd
    , sigma.mean, sigma.sd
    , prec.shape, prec.rate
    , prob.conc
    , gamma.lower, gamma.upper
    , a0.shape1, a0.shape2
    , a0.lower, a0.upper
    , trtname
    , compute.loglik, compute.ssc
  )
  # return(standat)
  sampler <- samplelist[[prior]]
  fit <- sampler$sample(data = standat, ...)
  attr(fit, 'prior') = prior
  attr(fit, 'standata') = standat
  attr(fit, 'dic') = NA
  attr(fit, 'pd') = NA
  ## Compute DIC if compute.loglik == TRUE
  if ( compute.loglik ) {
    parms.names      <- c('beta', 'sigma', 'log_lik')
    post.mean        <- fit$summary(parms.names, 'mean')
    postmean.loglik  <- post.mean$mean[grepl('log_lik\\[', post.mean$variable)]
    sigma            <- post.mean$mean[post.mean$variable == 'sigma']
    if (prior %in% c('leap', 'leap2'))
      sigma <- post.mean$mean[post.mean$variable == 'sigma[1]']
    beta             <- post.mean$mean[grepl('beta\\[', post.mean$variable)]
    loglik.postmean  <- dnorm(standat$y, standat$X %*% beta, sigma, log = TRUE)
    dev.postmean     <- -2 * sum(loglik.postmean)
    postmean.dev     <- -2 * sum(postmean.loglik)
    pd               <- postmean.dev - dev.postmean
    dic              <- pd + postmean.dev
    attr(fit, 'dic') <- dic
    attr(fit, 'pd')  <- pd
  }
  fit
}









#' Propensity score integrated power prior
#' 
#' Fit the propensity score integrated power prior
#' 
#' @param yname name of response variable
#' @param trtname name of treatment variable
#' @param ps.formula a right-sided formula giving the covariates to control for
#' @param data current data set
#' @param histdata historical data set
#' @param nstrata number of strata
#' @param mean.mean mean hyperparameter for normal prior on strata means
#' @param mean.sd sd hyperparameter for normal prior on strata mean
#' @param sigma.mean mean hyperparameter for half-normal prior on standard deviation
#' @param sigma.sd sd hyperparameter for half-normal prior on standard deviation
#' @param trim whether to trim the propensity scores. If `TRUE`, all observations from the historical data whose PSs lie outside the range of the current data will be excluded.
#' @param compute.loglik whether to compute log likelihood post sampling for leave-one-out and DIC
#' @param compute.ssc whether to compute the sample size contribution of the historical data
psipp <- function(
  yname, trtname, ps.formula, data, histdata
  , nstrata = 5
  , mean.mean = 0, mean.sd = 10
  , sigma.mean = 0, sigma.sd = 10
  , compute.loglik = FALSE, compute.ssc = FALSE
  , trim = TRUE
  , ...
) {
  if ( is.two.sided(ps.formula ) )
    stop('ps.formula should be a one-sided formula')
  
  ## Fit propensity score and conduct stratification (using psrwe package)
  data$studyid     <- 0
  histdata$studyid <- 1
  pooled    <- rbind(data, histdata) %>% 
    mutate(trt_n = as.numeric( as.factor( get(trtname) ) )) %>% 
    mutate(trt_n = trt_n - 1)
  ps.formula.full <- as.formula(paste('studyid', ps.formula))
  psrwe.fit <- psrwe_est(
    pooled %>% as.data.frame, ps.formula.full, v_arm = 'trt_n'
    , ctl_arm_level = 0, nstrata = nstrata, cur_grp_level = 0
  )
  
  ## Compute nominal number of patients to borrow (ntilde00)
  n00      <- pooled %>% filter(trt_n == 0, studyid == 1) %>% nrow   ## number of historical controls
  n10      <- pooled %>% filter(trt_n == 0, studyid == 0) %>% nrow   ## number of current controls
  n11      <- pooled %>% filter(trt_n == 1, studyid == 0) %>% nrow   ## number of current treated
  ntilde00 <- min(n00, n11 - n10)                                    ## nominal number of historical controls to borrow
  
  ## Compute a0 based on distance in PS distributions
  psrwe.dist <- psrwe_borrow(
    psrwe.fit, total_borrow = ntilde00, method = "distance"
  )
  a0     <- psrwe.dist$Borrow$Alpha
  a0[is.nan(a0)|is.na(a0)] <- 0
  pooled <- psrwe.fit$data %>%
    mutate(stratum = as.numeric(get('_strata_'))) %>%
    filter(complete.cases(.))
  
  ## Create stan data
  standat <- list(
      'n_trt' = pooled %>% filter(studyid == 0, trt_n == 1) %>% nrow
    , 'n_ctrl' = pooled %>% filter(studyid == 0, trt_n == 0) %>% nrow
    , 'n0_ctrl' = pooled %>% filter(studyid == 1, trt_n == 0) %>% nrow
    , 'J' = nstrata
    , 's_trt' = pooled %>% filter(studyid == 0, trt_n == 1 ) %>% select(stratum) %>% unlist
    , 's_ctrl' = pooled %>% filter(studyid == 0, trt_n == 0 ) %>% select(stratum) %>% unlist
    , 's0_ctrl' = pooled %>% filter(studyid == 1, trt_n == 0 ) %>% select(stratum) %>% unlist
    , 'y_trt' = pooled %>% filter(studyid == 0, trt_n == 1 ) %>% select(all_of(yname)) %>% unlist
    , 'y_ctrl' = pooled %>% filter(studyid == 0, trt_n == 0 ) %>% select(all_of(yname)) %>% unlist
    , 'y0_ctrl' = pooled %>% filter(studyid == 1, trt_n == 0 ) %>% select(all_of(yname)) %>% unlist
    , 'mean_mean' = mean.mean
    , 'mean_sd' = mean.sd
    , 'sigma_mean' = sigma.mean
    , 'sigma_sd' = sigma.sd
    , 'a0' = a0
    , 'compute_loglik' = as.numeric(compute.loglik)
    , 'compute_ssc' = as.numeric(compute.ssc)
  )
  fit <- psipp.stan$sample(data = standat, ...)
  attr(fit, 'prior') = 'psipp'
  attr(fit, 'standata') = standat
  attr(fit, 'dic') = NA
  attr(fit, 'pd') = NA
  ## Compute DIC if compute.loglik == TRUE
  if(compute.loglik) {
    parmnames       <- c('mean_trt', 'mean_ctrl', 'sigma', 'log_lik')
    post.mean       <- fit$summary('mean', variables = parmnames)
    mean_trt        <- post.mean$mean[grepl('mean_trt', post.mean$variable)]
    mean_ctrl       <- post.mean$mean[grepl('mean_ctrl', post.mean$variable)]
    sigma           <- post.mean$mean[grepl('sigma', post.mean$variable)]
    postmean.loglik <- post.mean$mean[grepl('log_lik', post.mean$variable)]
    postmean.mean   <- c( mean_trt[standat$s_trt], mean_ctrl[standat$s_ctrl] )
    postmean.sd     <- c(sigma[c(standat$s_trt, standat$s_ctrl)])
    y               <- c(standat$y_trt, standat$y_ctrl)
    loglik.postmean <- dnorm(y, mean = postmean.mean, sd = postmean.sd, log = TRUE)
    dev.postmean     <- -2 * sum(loglik.postmean)
    postmean.dev     <- -2 * sum(postmean.loglik)
    pd               <- postmean.dev - dev.postmean
    dic              <- pd + postmean.dev
    attr(fit, 'dic') <- dic
    attr(fit, 'pd')  <- pd
  }
  fit
}






