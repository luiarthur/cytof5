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
    "--b0PriorSd"
      arg_type = Float64
      default = 1.0
    "--b1PriorScale"
      arg_type = Float64
      default = 1.0
    "--b0TunerInit"
      arg_type = Float64
      default = 0.1
    "--b1TunerInit"
      arg_type = Float64
      default = 0.1
    "--fix_b0"
      arg_type = Bool
      default = false
    "--fix_b1"
      arg_type = Bool
      default = false
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
b0PriorSd = PARSED_ARGS["b0PriorSd"]
b1PriorScale = PARSED_ARGS["b1PriorScale"]
b0TunerInit = PARSED_ARGS["b0TunerInit"]
b1TunerInit = PARSED_ARGS["b1TunerInit"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
printFreq = PARSED_ARGS["printFreq"]
fix_b0 = PARSED_ARGS["fix_b0"]
fix_b1 = PARSED_ARGS["fix_b1"]
fix = Vector{Symbol}()

if fix_b0 fix=[fix; :b0] end
if fix_b1 fix=[fix; :b1] end


Random.seed!(SEED);
# END OF ARG PARSING

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

logger("Simulating Data ...");
Z = Cytof5.Model.genZ(J, K, 0.6)
dat = Cytof5.Model.genData(I, J, N, K, L, Z,
                           Dict(:b0=>-9.2, :b1=>2.3), # missMechParams
                           [0.2, 0.1, 0.3], # sig2
                           Dict(0=>-rand(L[0]) * 5,  # mus0
                                1=> rand(L[1]) * 5), # mus1
                           rand(K)*10, # a_W
                           Dict(z => rand(L[z])*10 for z in 0:1), # a_eta
                           sortLambda=false, propMissingScale=0.7)

y_dat = Cytof5.Model.Data(dat[:y])

logger("Generating priors ...");
@time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC,
                                        b0PriorSd=b0PriorSd,
                                        b1PriorScale=b1PriorScale)

logger("Generating initial state ...");
@time init = Cytof5.Model.genInitialState(c, y_dat)

logger("Fitting Model ...");
@time out, lastState, ll, metrics =
  Cytof5.Model.cytof5_fit(init, c, y_dat,
                          monitors=[[:Z, :lam, :W,
                                     :b0, :b1, :v,
                                     :sig2, :mus,
                                     :alpha, :v,
                                     :eta],
                                    [:y_imputed]],
                          fix=fix,
                          thins=[1, 100],
                          nmcmc=MCMC_ITER, nburn=BURN,
                          printFreq=printFreq,
                          b0_tune_init=b0TunerInit,
                          b1_tune_init=b1TunerInit,
                          computeLPML=true, computeDIC=true,
                          flushOutput=true)

logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out dat ll lastState c y_dat metrics

logger("MCMC Completed.");

#= Test
julia --color=yes sim.jl --I=3 --J=32 --N_factor=100 --K=8 --L=4 --K_MCMC=10 --L_MCMC=5 --RESULTS_DIR="bla" --EXP_NAME=small
=#
