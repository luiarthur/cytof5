using Cytof5, Random, RCall
using JLD2, FileIO

Random.seed!(10)
printDebug = false

I = 3
J = 32
N = [3, 1, 2] * 100 # Super fast even for 10000 obs. 
K = 4
L = 4

println("Simulating Data ...")
@time dat = Cytof5.Model.genData(I, J, N, K, L, sortLambda=true)
y_dat = Cytof5.Model.Data(dat[:y])

K_MCMC = 10
L_MCMC = 5

println("Generating priors ...")
@time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC)

println("Generating initial state ...")
@time init = Cytof5.Model.genInitialState(c, y_dat)

println("Fitting Model ...")
@time out, lastState, ll = Cytof5.Model.cytof5_fit(init, c, y_dat,
                                                   nmcmc=1000, nburn=10000,
                                                   numPrints=100)

println("Saving Data ...")
@save "result/out.jld2" out dat ll lastState

