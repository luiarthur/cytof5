println("Loading packages...") 
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
    "--MCMC_ITER"
      arg_type = Int
      required = true
    "--BURN"
      arg_type = Int
      required = true
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
    "--b0TunerInit"
      arg_type = Float64
      default = 0.1
    "--b1TunerInit"
      arg_type = Float64
      default = 0.1

    "--RESULTS_DIR"
      arg_type = String
      required = true
    "--EXP_NAME"
      arg_type = String
      required = true
    "--DATA_PATH"
      arg_type = String
      default = "data/cytof_cb_with_nan.jld2"
  end

  return parse_args(s)
end

println("Parsing command args ...") 
PARSED_ARGS = parse_cmd()
for (k,v) in PARSED_ARGS
  logger("$k => $v")
end

MCMC_ITER = PARSED_ARGS["MCMC_ITER"]
BURN = PARSED_ARGS["BURN"]
K_MCMC = PARSED_ARGS["K_MCMC"]
L_MCMC = PARSED_ARGS["L_MCMC"]
b0PriorSd = PARSED_ARGS["b0PriorSd"]
b1PriorScale = PARSED_ARGS["b1PriorScale"]
b0TunerInit = PARSED_ARGS["b0TunerInit"]
b1TunerInit = PARSED_ARGS["b1TunerInit"]
SEED = PARSED_ARGS["SEED"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
EXP_NAME = PARSED_ARGS["EXP_NAME"]
cbDataPath = PARSED_ARGS["DATA_PATH"]

Random.seed!(SEED);
# End of ArgParse

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

# Read CB Data
@load cbDataPath y_cb


dat = Cytof5.Model.Data(y_cb)

# MAIN
logger("Generating priors ...");
@time c = Cytof5.Model.defaultConstants(dat, K_MCMC, L_MCMC,
                                        b0PriorSd=b0PriorSd, b1PriorScale=b1PriorScale)

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
                                                   nmcmc=MCMC_ITER, nburn=BURN,
                                                   computeLPML=true, computeDIC=true,
                                                   b0_tune_init=b0TunerInit,
                                                   b1_tune_init=b1TunerInit,
                                                   printFreq=10, flushOutput=true)

logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out ll lastState c

logger("MCMC Completed.");
