
data {
  int<lower=0>      n_trt;
  int<lower=0>      n_ctrl;
  int<lower=0>      n0_ctrl;
  int<lower=0>      J;                      // number of strata
  int<lower=0,upper=J> s_trt[n_trt];        // strata membership for internal treated
  int<lower=0,upper=J> s_ctrl[n_ctrl];      // strata membership for internal control
  int<lower=0,upper=J> s0_ctrl[n0_ctrl];    // strata membership for external control
  vector[n_trt] y_trt;
  vector[n_ctrl] y_ctrl;
  vector[n0_ctrl] y0_ctrl;
  real mean_mean;
  real mean_sd;
  real sigma_mean;
  real<lower=0> sigma_sd;
  real<lower=0,upper=1> a0[J];
  int<lower=0,upper=1> compute_loglik;
  int<lower=0,upper=1> compute_ssc;
}
transformed data {
  int n = n_trt + n_ctrl;
  real ssc = 0;
  for ( i in 1:n0_ctrl ) {
    for ( j in 1:J ) {
      if ( s0_ctrl[i] == j )
        ssc += a0[j];
    }
  }
}
parameters {
  vector[J] mean_trt;
  vector[J] mean_ctrl;
  vector<lower=0>[J] sigma;
}

model {
  for ( j in 1:J ) {
    for ( i in 1:n_trt ) {
      if ( s_trt[i] == j )
        y_trt[i] ~ normal(mean_trt[j], sigma[j]);
    }
    for ( i in 1:n_ctrl ) {
      if ( s_ctrl[i] == j )
        y_ctrl[i] ~ normal(mean_ctrl[j], sigma[j]);
    }
    for ( i in 1:n0_ctrl ) {
      if ( s0_ctrl[i] == j )
        target += a0[j] * normal_lpdf(y0_ctrl[i] | mean_ctrl[j], sigma[j]);
    }
  }
  
  // initial priors
  mean_ctrl ~ normal(mean_mean, mean_sd);
  mean_trt  ~ normal(mean_mean, mean_sd);
  sigma     ~ normal(sigma_mean, sigma_sd);
}
generated quantities {
  real trteff = mean(mean_trt - mean_ctrl);
  vector[compute_loglik ? n : 0] log_lik;
  vector[compute_ssc ? 1 : 0] hist_ssc;
  if ( compute_loglik ) {
    int ctr = 0;
    for ( i in 1:n_trt ) {
      ctr += 1;
      for ( j in 1:J ) {
        if ( s_trt[i] == j )
          log_lik[ctr] = normal_lpdf(y_trt[i] | mean_trt[j], sigma[j]);
      }
    }
    for ( i in 1:n_ctrl ) {
      ctr += 1;
      for ( j in 1:J ) {
        if ( s_ctrl[i] == j )
          log_lik[ctr] = normal_lpdf(y_ctrl[i] | mean_ctrl[j], sigma[j]);
      }
    }
  }
  if (compute_ssc) {
    hist_ssc[1] = ssc;
  }
}
