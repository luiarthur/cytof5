println("Loading packages...")
@time begin
  using Cytof5
  using Random
  using JLD2, FileIO
  using ArgParse
end
println("Done loading packages.")

function logger(x; newline=true)
  if newline
    println(x)
  else
    print(x)
  end
  flush(stdout)
end

# ARG PARSING
function parse_cmd()
  s = ArgParseSettings()

  @add_arg_table s begin
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
    "--L"
      arg_type = Int
      required = true
    "--K_MCMC"
      arg_type = Int
      required = true
    "--L_MCMC"
      arg_type = Int
      required = true
    "--RESULTS_DIR"
      arg_type = String
      required = true
    "--b0PriorSd"
      arg_type = Float64
      default = 1.0
    "--b1PriorScale"
      arg_type = Float64
      default = 1.0
    "--SEED"
      arg_type = Int
      default = 0
    "--EXP_NAME"
      arg_type = String
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

I = PARSED_ARGS["I"]
J = PARSED_ARGS["J"]
N_factor = PARSED_ARGS["N_factor"]
N = N_factor * [3, 1, 2]
K = PARSED_ARGS["K"]
L = PARSED_ARGS["L"]
K_MCMC = PARSED_ARGS["K_MCMC"]
L_MCMC = PARSED_ARGS["L_MCMC"]
EXP_NAME = PARSED_ARGS["EXP_NAME"]
SEED = PARSED_ARGS["SEED"]
b0PriorSd = PARSED_ARGS["b0PriorSd"]
b1PriorScale = PARSED_ARGS["b1PriorScale"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]

Random.seed!(SEED);
# END OF ARG PARSING

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

logger("Simulating Data ...");
@time dat = Cytof5.Model.genData(I, J, N, K, L, sortLambda=false, useSimpleZ=false)
y_dat = Cytof5.Model.Data(dat[:y])

logger("Generating priors ...");
@time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC, b0PriorSd=b0PriorSd, b1PriorScale=b1PriorScale)

logger("Generating initial state ...");
@time init = Cytof5.Model.genInitialState(c, y_dat)

logger("Fitting Model ...");
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
                                                   printFreq=10, computeLPML=true,
                                                   flushOutput=true)

logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out dat ll lastState c y_dat

logger("MCMC Completed.");

#= Test
julia sim.jl --I=3 --J=32 --N_factor=100 --K=8 --L=4 --K_MCMC=10 --L_MCMC=5 --RESULTS_DIR="bla" --EXP_NAME=small
=#
