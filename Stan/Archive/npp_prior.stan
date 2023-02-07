
data {
  int<lower=0> n0;
  int<lower=0> p;
  vector[n0] y0;
  matrix[n0,p] X0;
  real<lower=0> gamma0;
  real<lower=0> delta0;
  real<lower=0> a0_lower;
  real<lower=0> a0_upper;
}
transformed data {
  vector[p] betahat0 = mdivide_left_spd( crossprod(X0), X0' * y0 );
  real sse0 = dot_self(y0 - X0 * betahat0);
  real half_gamma0 = gamma0 / 2.0;
  real half_delta0 = delta0 / 2.0;
  real logC0 = -0.5 * p * log(2*pi()) - 0.5 * log_determinant(crossprod(X0));
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
  real error0 = dot_self(y0 - X0 * beta);
  real a0n0 = a0 * n0;
  real log_prec = log(prec);
  real shape_post = 0.5 * (a0n0 + delta0 - p);
  real rate_post = 0.5 * (a0*sse0 + gamma0);
  real log_a0 = log(a0);
  real logC = -0.5 * p * log_a0 + lgamma(shape_post) - shape_post * log(rate_post);
  target +=   (0.5 * (a0n0 + delta0) - 1) * log_prec 
            - 0.5 * a0 * prec * ( error0 + gamma0 * inv(a0) );
  target += -(logC + logC0);
}

