println("Loading packages...") 
using Cytof5
using Random, Distributions
using JLD2, FileIO
using ArgParse
include("PreProcess.jl")
include("post_process_defs.jl")

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

#{{{1
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
    "--L0_MCMC"
      arg_type = Int
      required = true
    "--L1_MCMC"
      arg_type = Int
      required = true
    "--tau0"
      arg_type = Float64
      default = 0.0
    "--tau1"
      arg_type = Float64
      default = 0.0
    "--SEED"
      arg_type = Int
      default = 0
    "--subsample"
      arg_type = Float64
      default = 0.0
    "--USE_REPULSIVE"
      arg_type = Bool
      default = false

    "--RESULTS_DIR"
      arg_type = String
      required = true
    "--EXP_NAME"
      arg_type = String
      required = true
    "--DATA_PATH"
      arg_type = String
      required=true
  end

  return parse_args(s)
end
#}}}1

println("Parsing command args ...") 
PARSED_ARGS = parse_cmd()
for (k,v) in PARSED_ARGS
  Cytof5.Model.logger("$k => $v")
end

MCMC_ITER = PARSED_ARGS["MCMC_ITER"]
BURN = PARSED_ARGS["BURN"]
K_MCMC = PARSED_ARGS["K_MCMC"]
L_MCMC = Dict(0 => PARSED_ARGS["L0_MCMC"], 1 => PARSED_ARGS["L1_MCMC"])
TAU0= PARSED_ARGS["tau0"]
TAU1= PARSED_ARGS["tau1"]
SEED = PARSED_ARGS["SEED"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
EXP_NAME = PARSED_ARGS["EXP_NAME"]
USE_REPULSIVE = PARSED_ARGS["USE_REPULSIVE"]
cbDataPath = PARSED_ARGS["DATA_PATH"]
cbDataPath = PARSED_ARGS["DATA_PATH"]
subsample = PARSED_ARGS["subsample"]

Random.seed!(SEED);
# End of ArgParse

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

# Read CB Data
cbData = loadSingleObj(cbDataPath)
# Reduce Data by removing highly non-expressive / expressive columns
goodColumns, J = PreProcess.preprocess!(cbData, maxNanOrNegProp=.9, maxPosProp=.9,
                                        subsample=subsample, rowThresh=-6.0)
Cytof5.Model.logger("good columns: $goodColumns")

# Possibly reduce data size
Cytof5.Model.logger(size.(cbData))

# Save Reduced Data
mkpath("$(OUTDIR)/reduced_data")
@save "$(OUTDIR)/reduced_data/reduced_cb.jld2" deepcopy(cbData)

# Create Data Object
dat = Cytof5.Model.Data(cbData)

# MAIN
Cytof5.Model.logger("\nGenerating priors ...");
# sig2_a, sig2_b = Cytof5.Model.solve_ig_params(mu=.2, sig2=.01)
@time c = Cytof5.Model.defaultConstants(dat, K_MCMC, L_MCMC,
                                        tau0=TAU0, tau1=TAU1,
                                        # sig2_prior=InverseGamma(sig2_a, sig2_b),
                                        # sig2_range=[0.0, 10.0],
                                        sig2_prior=InverseGamma(3.0, 2.0),
                                        mus0_range=[-15.0, 0.0],
                                        mus1_range=[0.0, 10.0],
                                        alpha_prior=Gamma(0.1, 10.0),
                                        yQuantiles=[.1, .25, .4], pBounds=[.05, .8, .05],
                                        eps=.2, sig2_0=10.0)

# Plot missing mechanism
util.plotPdf("$(OUTDIR)/prob_miss.pdf")
R"par(mfrow=c($(dat.I), 1))"
for i in 1:dat.I
  util.plotProbMiss(c.beta, i)
end
R"par(mfrow=c(1,1))"
util.devOff()

Cytof5.Model.logger("\nGenerating initial state ...");
# @time init = Cytof5.Model.genInitialState(c, dat)
@time init = Cytof5.Model.smartInit(c, dat)

Cytof5.Model.logger("Fitting Model ...");
@time out, lastState, ll, metrics =
  Cytof5.Model.cytof5_fit(init, c, dat,
                          monitors=[[:Z, :lam, :W,
                                     :sig2, :mus,
                                     :alpha, :v,
                                     :eta],
                                    [:y_imputed, :gam]],
                           thins=[1, round(Int, MCMC_ITER / 10)],
                           nmcmc=MCMC_ITER, nburn=BURN,
                           computeLPML=true, computeDIC=true,
                           use_repulsive=USE_REPULSIVE,
                           printFreq=10, flushOutput=true)

Cytof5.Model.logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out ll lastState c metrics
@save "$(OUTDIR)/reduced_data/reduced_cb.jld2" deepcopy(cbData)

Cytof5.Model.logger("MCMC Completed.");
