println("Loading packages...") 
using Cytof5, Random
using JLD2, FileIO

dirContains(d::String, file::String) = file in split(read(`ls $d`, String))

RESULTS_PATH = "results/flowSearch/"
DATA_PATHS = map(x -> "$RESULTS_PATH/$x", split(read(`ls $RESULTS_PATH`, String)))
DATA_PATHS = filter(d -> dirContains(d, "dat.jld2"), DATA_PATHS)

MCMC_ITER = 1000
BURN = 10000
K_MCMC = 10
L_MCMC = 5
b0PriorSd = 0.1
b1PriorScale = 0.1

function sim(path)
  @load "$(path)/dat.jld2" dat
  d = Cytof5.Model.Data(dat[:y])
  
  println("Generating Priors ...")
  @time c = Cytof5.Model.defaultConstants(d, K_MCMC, L_MCMC,
                                          b0PriorSd=b0PriorSd, b1PriorScale=b1PriorScale)

  println("Generating initial state ...")
  @time init = Cytof5.Model.genInitialState(c, d)

  println("Fitting Model ...");
  @time out, lastState, ll, metrics =
    Cytof5.Model.cytof5_fit(init, c, d,
                            monitors=[[:Z, :lam, :W,
                                       :b0, :b1, :v,
                                       :sig2, :mus,
                                       :alpha, :v,
                                       :eta],
                                      [:y_imputed]],
                            thins=[1, 100],
                            nmcmc=MCMC_ITER, nburn=BURN,
                            computeDIC=true, computeLPML=true,
                            printFreq=10,
                            flushOutput=true)

  println("Saving Data ...")
  @save "$(path)/mcmcout.jld2" out ll lastState c dat metrics;

  println("MCMC Completed.")
end

for path in DATA_PATHS
  println(path)
  sim(path)
end 
