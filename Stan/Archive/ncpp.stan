
data {
  int<lower=0> n;
  int<lower=0> n0;
  int<lower=0> p;
  vector[n] y;
  vector[n0] y0;
  matrix[n,p] X;
  matrix[n0,p] X0;
  real beta_mean;
  real<lower=0> beta_sd;
  real<lower=0> prec_shape;
  real<lower=0> prec_rate;
  real<lower=0> a0_shape1;
  real<lower=0> a0_shape2;
  real<lower=0,upper=1> a0_lower;
  real<lower=a0_lower,upper=1> a0_upper;
  int X0arr[p-1];
  int trtidx;
}
transformed data {
  matrix[n0,p-1] X02 = X0[,X0arr];
  matrix[p-1,p-1] X02tX02 = crossprod(X02);
  vector[p-1] betahat02 = mdivide_left_spd(X02tX02, X02' * y0);
}
parameters {
  vector[p] beta;
  real<lower=0> prec;
  real<lower=a0_lower,upper=a0_upper> a0;
}
transformed parameters {
  real sigma = inv_sqrt(prec);
}
model {
  y ~ normal_id_glm(X, 0, beta, sigma);
  target += multi_normal_prec_lpdf(beta[X0arr] | betahat02, a0 * prec * X02tX02);
  target += normal_lpdf(beta[trtidx] | beta_mean, beta_sd);
  target += beta_lpdf(a0 | a0_shape1, a0_shape2);
}
generated quantities {
  real effsize = n + a0 * n0;
}

