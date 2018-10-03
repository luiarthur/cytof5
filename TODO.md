# TODO
- [x] Stop using missing (bloats storage by 10x)
- [ ] FlowSOM comparison
- [ ] Change manuscript
- [ ] Generate small data with non-equidistant mu and different sig2
- [ ] strong prior for b0, b1
    - [ ] different tuning parameter for b1?
    - [ ] different prior for b1?
    - [ ] Refer to AMCMC paper
        - [ ] change M
        - [ ] change `delta(n) = min(.01, 1/sqrt(n))` to something else
- [ ] Try different values for K_MCMC = [6, 7, 8, 9, 10, 11] with L_TRUE=5 and K_TRUE=8
    - we expect that ll should increase from K_MCMC=6 to K_MCMC=8, then plateau.
- [ ] DIC / LPML for model comparison
    - DIC can be found in BDA3
    - LPML can be found in paper by Dey, Gelfand
    - Should the computation be using `m_inj` as well? Specifically,
      `likelihood= prod(Normal(observed y_inj | params)) * prod(Bern(m_inj | p_inj))`
