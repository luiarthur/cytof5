# TODO
- [ ] Draft a paper
    - [ ] pick one of the simulation study results (sim 64 or 98) to include
    - [ ] Intro (blank for now)
    - [ ] probability model (with new changes)
        - [ ] rewrite section on prior for Z using traditional IBP construction
              (instead of probit since we are not using correlation between
              markers)
    - [ ] sim study (LPML / DIC for several K)
    - [ ] CB study  (LPML / DIC for several K)
    - [ ] compare to FlowSOM?
    - [ ] preprocessing by removing markers?
    - [ ] Conclusions (blank for now)
    - Eventually:
        - [ ] sim study using other simulation data
- [x] strong prior for b0, b1
    - [x] different tuning parameter for b1?
    - [x] different prior for b1?
    - [x] Refer to AMCMC paper
        - [ ] change M
        - [ ] change `delta(n) = min(.01, 1/sqrt(n))` to something else
- [ ] Use `Plots.jl` with `pyplot` back-end
- [ ] Check the repulsive implementations
- [ ] Use smaller field-types in State
    - [ ] `Float64` => `Float32` for y?
    - [ ] `Int` => `Int8` for indicators like gamma and Z
- [ ] investigate why `loglike` and `DIC / LPML` is `-infty` sometimes

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
