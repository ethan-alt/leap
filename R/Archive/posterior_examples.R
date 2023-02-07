## source wrapper functions
source('/proj/ibrahimlab/leap/R/leap_wrappers.R')
library(plyr)
library(dplyr)
library(loo)
library(tables)
library(kableExtra)


n = 100
n0 = 50
data <- data.frame(a = rbinom(n, 1, 2/3), x = rnorm(n), studyid = 1)
histdata <- data.frame(a = 0, x = rnorm(n0), studyid = 0)
data$y     <- rnorm(n, 2 + 5 * data$a - data$x, sd = 5)
histdata$y <- rnorm(n0, 2 + 5 * histdata$a - histdata$x, sd = 5)
ps.formula <- ~ x
formula <- y ~ a + x


nwarmup   <- 2000
nsamples  <- 50000
parchains <- 4
niter     <- ceiling(nsamples / parchains)
##------------------------
## Fit PS integrated pp
##------------------------
fit.psipp <- psipp(
  yname, trtname, ps.formula, data, histdata
  , compute.loglik = TRUE, compute.ssc = TRUE
  , iter_warmup = nwarmup, iter_sampling = niter
  , parallel_chains = parchains, chains = parchains
)
summ.psipp <- fit.psipp$summary(
  c('trteff', 'hist_ssc'), c('mean', 'sd', 'ess_bulk', 'quantile' = ~quantile2(., probs = c(0.025, 0.975)))
)
summ.psipp$prior <- 'psipp'



## Fit leap with K = 2 components
fit.leap2 <- get.samples(
  formula, data, histdata, prior = 'leap'
  , prob.conc = c(1, 1) ## uniform over the 2-dim simplex
  , compute.loglik = TRUE, compute.ssc = TRUE
  , iter_warmup = nwarmup, iter_sampling = niter
  , parallel_chains = parchains, chains = parchains
  , adapt_delta = 0.90
)
summ.leap2 <- fit.leap2$summary(
  c('beta[2]', 'gamma', 'hist_ssc'), c('mean', 'sd', 'ess_bulk', 'quantile' = ~quantile2(., probs = c(0.025, 0.975)))
)
summ.leap2$prior <- 'leap2'


## Fit leap with K = 3 components
fit.leap3 <- get.samples(
  formula, data, histdata, prior = 'leap'
  , prob.conc = c(1, 1, 1) ## uniform over the 3-dim simplex
  , compute.loglik = TRUE, compute.ssc = TRUE
  , iter_warmup = nwarmup, iter_sampling = niter
  , parallel_chains = parchains, chains = parchains
  , adapt_delta = 0.90
)
summ.leap3 <- fit.leap3$summary(
  c('beta[2]', 'gamma', 'hist_ssc'), c('mean', 'sd', 'ess_bulk', 'quantile' = ~quantile2(., probs = c(0.025, 0.975)))
)
summ.leap3$prior <- 'leap3'






## Fit normalized partial borrowing power prior (pbnpp). Must specify name of treatment variable
fit.pbnpp <- get.samples(
  formula, data, histdata, prior = 'pbnpp'
  , compute.loglik = T, compute.ssc = T
  , iter_warmup = nwarmup, iter_sampling = niter
  , parallel_chains = parchains, chains = parchains
  , trtname = trtname
  , adapt_delta = 0.90
)
summ.pbnpp <- fit.pbnpp$summary(
  c('beta[2]', 'a0', 'hist_ssc'), c('mean', 'sd', 'ess_bulk', 'quantile' = ~quantile2(., probs = c(0.025, 0.975)))
)
summ.pbnpp$prior <- 'pbnpp'




## Fit reference prior
fit.refprior <- get.samples(
  formula, data, histdata, prior = 'refprior'
  , compute.loglik = TRUE
  , iter_warmup = nwarmup, iter_sampling = niter, parallel_chains = parchains, chains = parchains
)
summ.refprior <- fit.refprior$summary(
  'beta[2]', c('mean', 'sd', 'ess_bulk', 'quantile' = ~quantile2(., probs = c(0.025, 0.975)))
)
summ.refprior$prior <- 'refprior'


## put all into one data frame
summ <- rbind.fill(summ.leap2, summ.leap3, summ.psipp, summ.pbnpp, summ.refprior)

## Put all objects into list
fitlist <- list(
  leap2 = fit.leap2, leap3 = fit.leap3, psipp = fit.psipp
  , pbnpp = fit.pbnpp, refprior = fit.refprior
)

## Obtain dic and number of effective params (dic)
dic <- sapply(fitlist, function(x) attr(x, 'dic'))
pd  <- sapply(fitlist, function(x) attr(x, 'pd'))
dic <- data.frame(dic, pd)
dic$prior <- rownames(dic)
summ <- summ %>% join(dic, by = 'prior')

## Obtain leave-one-out results
loo.list <- lapply(fitlist, function(x) x$loo())
loo.comp <- loo_compare(loo.list)
loo.comp <- data.frame(loo.comp)
loo.comp$prior <- rownames(loo.comp)


## Put everything together
summ <- summ %>% 
  join(loo.comp, by = 'prior') %>%
  join(dic, by = 'prior') %>%
  arrange(prior) %>%
  relocate(prior)

## Rename beta[2] --> trteff
summ$variable <- ifelse(summ$variable == 'beta[2]', 'trteff', summ$variable)

## Create table
tab <- tabular(
  Factor(prior, 'Prior', levelnames = c('Leap (K=2)', 'Leap (K=3)', 'PBNPP', 'PSIPP', 'Reference')) ~
    Heading() * identity * (
      Format(digits = 5) * Heading('DIC') * dic + Heading('LOO-IC') * looic + 
        Factor(variable, '', levelnames = c('Hist. SSC', 'Treatment')) * (
          Heading('Mean') * mean + Heading('Std. Dev.') * sd)
        )
  , data = summ %>% 
      mutate_if(is.numeric, round, digits = 3) %>%
      filter(variable %in% c('trteff', 'hist_ssc[1]'))
)
tab

## Latex table
tab.latex <- tab %>% toLatex()

## Using kableExtra package
tab.kbl <- tab %>%
  toKable()

