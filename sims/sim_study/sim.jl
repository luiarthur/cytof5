println("Loading packages...")
@time begin
  using Cytof5
  using Random
  using JLD2, FileIO
  using ArgParse
  import Cytof5.Model.logger
end
println("Done loading packages.")


# ARG PARSING
function parse_cmd()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--MCMC_ITER"
      arg_type = Int
      default = 1000
    "--BURN"
      arg_type = Int
      default = 10000
    "--I"
      arg_type = Int
      required = true
    "--J"
      arg_type = Int
      required = true
    "--N_factor"
      arg_type = Int
      required = true
    "--K"
      arg_type = Int
      required = true
    "--K_MCMC"
      arg_type = Int
      required = true
    "--L0"
      arg_type = Int
      required = true
    "--L1"
      arg_type = Int
      required = true
    "--L0_MCMC"
      arg_type = Int
      required = true
    "--L1_MCMC"
      arg_type = Int
      required = true
    "--RESULTS_DIR"
      arg_type = String
      required = true
    "--SEED"
      arg_type = Int
      default = 0
    "--EXP_NAME"
      arg_type = String
    "--printFreq"
      arg_type = Int
      default = 50
  end

  PARSED_ARGS = parse_args(s)

  if PARSED_ARGS["EXP_NAME"] == nothing
    logger("EXP_NAME not defined. Making default experiment name.")
    MAIN_ARGS = filter(d -> !(d.first in ("RESULTS_DIR", "EXP_NAME")), PARSED_ARGS)
    PARSED_ARGS["EXP_NAME"] = join(["$k$v" for (k, v) in MAIN_ARGS], '_')
  end

  return PARSED_ARGS
end

PARSED_ARGS = parse_cmd()
for (k,v) in PARSED_ARGS
  logger("$k => $v")
end

MCMC_ITER = PARSED_ARGS["MCMC_ITER"]
BURN = PARSED_ARGS["BURN"]
I = PARSED_ARGS["I"]
J = PARSED_ARGS["J"]
N_factor = PARSED_ARGS["N_factor"]
N = N_factor * [3, 1, 2]
K = PARSED_ARGS["K"]
K_MCMC = PARSED_ARGS["K_MCMC"]

L0 = PARSED_ARGS["L0"]
L1 = PARSED_ARGS["L1"]
L = Dict(0 => L0, 1 => L1)
L0_MCMC = PARSED_ARGS["L0_MCMC"]
L1_MCMC = PARSED_ARGS["L1_MCMC"]
L_MCMC = Dict(0 => L0_MCMC, 1 => L1_MCMC)

EXP_NAME = PARSED_ARGS["EXP_NAME"]
SEED = PARSED_ARGS["SEED"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
printFreq = PARSED_ARGS["printFreq"]


Random.seed!(SEED);
# END OF ARG PARSING

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

logger("Simulating Data ...");
Z = Cytof5.Model.genZ(J, K, 0.6)
dat = Cytof5.Model.genData(J=J, N=N, K=K, L=L, Z=Z,
                           beta=[-9.2, -2.3],
                           sig2=[0.2, 0.1, 0.3],
                           mus=Dict(0=>-rand(L[0]) * 5,
                                    1=> rand(L[1]) * 5),
                           a_W=rand(K)*10,
                           a_eta=Dict(z => rand(L[z])*10 for z in 0:1),
                           sortLambda=false, propMissingScale=0.7)

y_dat = Cytof5.Model.Data(dat[:y])

logger("Generating priors ...");
@time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC)
Cytof5.Model.printConstants(c)

logger("Generating initial state ...");
@time init = Cytof5.Model.genInitialState(c, y_dat)

logger("Fitting Model ...");
@time out, lastState, ll, metrics =
  Cytof5.Model.cytof5_fit(init, c, y_dat,
                          monitors=[[:Z, :lam, :W,
                                     :sig2, :mus,
                                     :alpha, :v,
                                     :eta],
                                    [:y_imputed]],
                          thins=Int[1, MCMC_ITER / 10],
                          nmcmc=MCMC_ITER, nburn=BURN,
                          printFreq=printFreq,
                          computeLPML=true, computeDIC=true,
                          flushOutput=true)

logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out dat ll lastState c y_dat metrics

logger("MCMC Completed.");

#= Test
julia --color=yes sim.jl --I=3 --J=32 --N_factor=100 --K=8 --L=4 --K_MCMC=10 --L_MCMC=5 --RESULTS_DIR="bla" --EXP_NAME=small
=#
