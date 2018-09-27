import Pkg
Pkg.activate("../../")

using Cytof5, Random, RCall
using JLD2, FileIO

Random.seed!(10);
printDebug = false;

println(ARGS)
I = parse(Int, ARGS[1]) # 3
J = parse(Int, ARGS[2]) # 32
N_factor = parse(Int, ARGS[3]) # 100
N = N_factor * [3, 1, 2]
K = parse(Int, ARGS[4]) # 4
L = parse(Int, ARGS[5]) # 4

OUTDIR = "result/N$(N_factor)/"
mkpath(OUTDIR)

println("Simulating Data ..."); flush(stdout)
@time dat = Cytof5.Model.genData(I, J, N, K, L, sortLambda=false, useSimpleZ=false)
y_dat = Cytof5.Model.Data(dat[:y])

K_MCMC = parse(Int, ARGS[6]) # 10
L_MCMC = parse(Int, ARGS[7]) # 5

println("Generating priors ..."); flush(stdout)
@time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC)

println("Generating initial state ..."); flush(stdout)
@time init = Cytof5.Model.genInitialState(c, y_dat)

println("Fitting Model ..."); flush(stdout)
@time out, lastState, ll = Cytof5.Model.cytof5_fit(init, c, y_dat,
                                                   monitors=[[:Z, :lam, :W,
                                                              :b0, :b1, :v,
                                                              :sig2, :mus,
                                                              :alpha, :v,
                                                              :eta],
                                                             [:y_imputed]],
                                                   thins=[1, 100],
                                                   nmcmc=1000, nburn=10000,
                                                   #nmcmc=2, nburn=2,
                                                   numPrints=100,
                                                   flushOutput=true)

println("Saving Data ...")
@save "$(OUTDIR)/N$(N_factor).jld2" out dat ll lastState c y_dat

println("MCMC Completed.")
