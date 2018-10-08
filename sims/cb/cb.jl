import Pkg
Pkg.activate("../../")

using Cytof5, Random
using JLD2, FileIO
using ArgParse

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
    "--K_MCMC"
      arg_type = Int
      required = true
    "--L_MCMC"
      arg_type = Int
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

    "--RESULTS_DIR"
      arg_type = String
      required = true
    "--EXP_NAME"
      arg_type = String
      required = true
  end

  return parse_args(s)
end

PARSED_ARGS = parse_cmd()
for (k,v) in PARSED_ARGS
  logger("$k => $v")
end

K_MCMC = PARSED_ARGS["K_MCMC"]
L_MCMC = PARSED_ARGS["L_MCMC"]
b0PriorSd = PARSED_ARGS["b0PriorSd"]
b1PriorScale = PARSED_ARGS["b1PriorScale"]
SEED = PARSED_ARGS["SEED"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
EXP_NAME = PARSED_ARGS["EXP_NAME"]

Random.seed!(SEED);
# End of ArgParse

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

# Read CB Data
cbDataPath = "data/cytof_cb.jld2"
@load cbDataPath y_cb
dat = Cytof5.Model.Data(y_cb)

# MAIN
logger("Generating priors ...");
@time c = Cytof5.Model.defaultConstants(dat, K_MCMC, L_MCMC, b0PriorSd=b0PriorSd, b1PriorScale=b1PriorScale)

logger("Generating initial state ...");
@time init = Cytof5.Model.genInitialState(c, dat)

logger("Fitting Model ...");
@time out, lastState, ll = Cytof5.Model.cytof5_fit(init, c, dat,
                                                   monitors=[[:Z, :lam, :W,
                                                              :b0, :b1, :v,
                                                              :sig2, :mus,
                                                              :alpha, :v,
                                                              :eta],
                                                             [:y_imputed]],
                                                   thins=[1, 100],
                                                   nmcmc=1000, nburn=15000,
                                                   #nmcmc=2, nburn=2,
                                                   printFreq=10,
                                                   flushOutput=true)

logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out ll lastState c

logger("MCMC Completed.");
