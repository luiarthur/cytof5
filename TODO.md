# TODO
- [x] strong prior for b0, b1
    - [x] different tuning parameter for b1?
    - [x] different prior for b1?
    - [x] Refer to AMCMC paper
        - [ ] change M
        - [ ] change `delta(n) = min(.01, 1/sqrt(n))` to something else
- [x] FlowSOM detective work:
    - [x] Simulate data until FlowSOM breaks
    - [x] Then, do MCMC
- [ ] revise manuscript
- [ ] strategically pre-process data
    - [ ] if across all samples, a marker has > 90% missing or negative, then remove
    - [ ] if across all samples, a marker has > 90% positive, then remove
- [ ] Use `Plots.jl` with `pyplot` backend
- [x] Stop using missing (bloats storage by 10x)
- [x] FlowSOM comparison
- [x] Generate small data with non-equidistant mu and different sig2
- [x] replace numPrints with printFreq
- [x] Try different values for K_MCMC = [6, 7, 8, 9, 10, 11] with L_TRUE=5 and K_TRUE=8
    - we expect that ll should increase from K_MCMC=6 to K_MCMC=8, then plateau.
- [x] DIC / LPML for model comparison
    - DIC can be found in BDA3
    - LPML can be found in paper by Dey, Gelfand
    - Should the computation be using `m_inj` as well? Specifically,
      `likelihood= prod(Normal(observed y_inj | params)) * prod(Bern(m_inj | p_inj))`
- [ ] Write a simple Julia linter in Python3
    - [ ] catch usage of undefined variables within scope
        - [ ] e.g. catch errors like this:
        ``` julia
        function f(x::Int)::Int
          return x + y # y is not defined!
        end
        ```
    - [ ] catch bad operations on different types
        - [ ] e.g. catch `1 + "xyz"`
    - [ ] check for function argument types
        - e.g.
        ```julia
        f(x::Int) = x + 1

        # This should be caught:
        f("xyz") # incorrect argument!
        ```
    - [ ] check existence of fields in a class object
        - e.g.
        ```julia
        struct Bob
          x::Int
          y::Float64
        end

        #= This should be caught:
        b = Bob("a", 2.0) # x should be Int!
        =# 

        b = Bob(2, 4.0)

        # This should be caught:
        b.z # There is no field z!
        ```
    - [ ] Look into imported libraries

