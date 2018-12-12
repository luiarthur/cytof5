# TODO
- [ ] run `rgrep lam | grep update`
- [x] update lam
- [x] update W
- [ ] update y_imputed
- [x] update mus
- [x] update Z
- [ ] update sig2
- [x] update gam
- [x] update eta
- [x] update logdnoisy
- [x] `mclust` package in R to set initial values
- [ ] visually check to see if mu* are reasonably initialized according to
      `psi` and `tau^2`
- [ ] FOR NOW, KEEP instead of throw away cells with `y < -6`
- [ ] FOR NOW, DO NOT remove markers with `> 10` missing markers
- [ ] plot missing mechanism
- [x] change the update order
- [x] For log-like, LPML, DIC, don't compute `p(miss=1 | y_obs)`
- [x] keep `tau^2` empirically determined, and not artificially small
- [x] quadratic missing mechanism
- [x] `qqplots`

# Not Urgent
- [ ] Use `PyPlot.jl` for plots
- [ ] Check the repulsive implementations
    - [ ] for sim study
        - [ ] start with true Z having K (=4?) columns, where column `k`is
              different from column 1 by exactly `k-1` markers, for 
              `k=2,...,K`. In addition, the other columns `k=2,...,K` should
              be "quite different" from each other. (Perhaps, having a distance
              of at least 3.)
        - [ ] change the prior and see how the posterior Z est changes.
        - [ ] the true W matrix should place most weight on column 1, but vary
              this setting to see what happens
- [ ] Use smaller field-types in State
    - [ ] `Float64` => `Float32` for y?
    - [ ] `Int` => `Int8` for indicators like gamma and Z

# DONE
- [x] FlowSOM detective work:
    - [x] Simulate data until FlowSOM breaks
    - [x] Then, do MCMC
- [x] revise manuscript
- [x] strategically pre-process data
    - [x] if across all samples, a marker has > 90% missing or negative, then remove
    - [x] if across all samples, a marker has > 90% positive, then remove
- [x] Stop using missing (bloats storage by 10x)
- [x] FlowSOM comparison
- [x] Generate small data with non-equidistant mu and different `sig2`
- [x] replace `numPrints` with `printFreq`
- [x] Try different values for K_MCMC = [6, 7, 8, 9, 10, 11] with L_TRUE=5 and K_TRUE=8
    - we expect that ll should increase from K_MCMC=6 to K_MCMC=8, then plateau.
- [x] DIC / LPML for model comparison
    - DIC can be found in BDA3
    - LPML can be found in paper by Dey, Gelfand
    - Should the computation be using `m_inj` as well? Specifically,
      `likelihood= prod(Normal(observed y_inj | params)) * prod(Bern(m_inj | p_inj))`
- [x] use macro to generate named constructor for any `struct`
- [x] fixing b0, b1
  - [x] strong prior for b0, b1
      - [x] different tuning parameter for b1?
      - [x] different prior for b1?
      - [x] Refer to AMCMC paper
          - [ ] change M
          - [ ] change `delta(n) = min(.01, 1/sqrt(n))` to something else
- [x] investigate why `loglike` and `DIC / LPML` is `-infty` sometimes
    - [x] reason was that observed y were too negative to be observed according to the
          missing mechanism. In response, we are fixing the missing mechanism and
          treating the missing mechanism parts as constants in likelihood.


# Notes:
- when working on git branches for solo projects, this one liner can push all branches
  to the remote
  ```bash
  git push --all -u

  # this will fetch branch from remote
  git fetch
  git checkout <mybranch>
  ```
