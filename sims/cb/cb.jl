println("Loading packages...") 
using Cytof5
using Random, Distributions
using JLD2, FileIO
using ArgParse
include("PreProcess.jl")

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
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
    "--tau0"
      arg_type = Float64
      default = 0.0
    "--tau1"
      arg_type = Float64
      default = 0.0
    "--SEED"
      arg_type = Int
      default = 0
    "--b0TunerInit"
      arg_type = Float64
      default = 0.1
    "--b1TunerInit"
      arg_type = Float64
      default = 0.1
    "--fix_b0"
      arg_type = Bool
      default = true
    "--fix_b1"
      arg_type = Bool
      default = true
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

println("Parsing command args ...") 
PARSED_ARGS = parse_cmd()
for (k,v) in PARSED_ARGS
  Cytof5.Model.logger("$k => $v")
end

MCMC_ITER = PARSED_ARGS["MCMC_ITER"]
BURN = PARSED_ARGS["BURN"]
K_MCMC = PARSED_ARGS["K_MCMC"]
L_MCMC = PARSED_ARGS["L_MCMC"]
b0PriorSd = PARSED_ARGS["b0PriorSd"]
b1PriorScale = PARSED_ARGS["b1PriorScale"]
TAU0= PARSED_ARGS["tau0"]
TAU1= PARSED_ARGS["tau1"]
b0TunerInit = PARSED_ARGS["b0TunerInit"]
b1TunerInit = PARSED_ARGS["b1TunerInit"]
SEED = PARSED_ARGS["SEED"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
EXP_NAME = PARSED_ARGS["EXP_NAME"]
USE_REPULSIVE = PARSED_ARGS["USE_REPULSIVE"]
cbDataPath = PARSED_ARGS["DATA_PATH"]
cbDataPath = PARSED_ARGS["DATA_PATH"]
fix_b0 = PARSED_ARGS["fix_b0"]
fix_b1 = PARSED_ARGS["fix_b1"]
fix = Vector{Symbol}()

if fix_b0 fix=[fix; :b0] end
if fix_b1 fix=[fix; :b1] end


Random.seed!(SEED);
# End of ArgParse

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

# Read CB Data
cbData = loadSingleObj(cbDataPath)
# Reduce Data by removing highly non-expressive / expressive columns
goodColumns, J = PreProcess.preprocess!(cbData, maxNanOrNegProp=.9, maxPosProp=.9)
Cytof5.Model.logger(size.(cbData))

# Save Reduced Data
mkpath("$(OUTDIR)/reduced_data/")

# Create Data Object
dat = Cytof5.Model.Data(cbData)

# MAIN
Cytof5.Model.logger("\nGenerating priors ...");
@time c = Cytof5.Model.defaultConstants(dat, K_MCMC, L_MCMC,
                                        tau0=TAU0, tau1=TAU1,
                                        b0PriorSd=b0PriorSd, b1PriorScale=b1PriorScale)
Cytof5.Model.printConstants(c)

Cytof5.Model.logger("\nGenerating initial state ...");
@time init = Cytof5.Model.genInitialState(c, dat)

Cytof5.Model.logger("Fitting Model ...");
@time out, lastState, ll, metrics =
  Cytof5.Model.cytof5_fit(init, c, dat, monitors=[[:Z, :lam, :W,
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
                           fix=fix,                           
                           use_repulsive=USE_REPULSIVE,
                           printFreq=10, flushOutput=true)

Cytof5.Model.logger("Saving Data ...");
@save "$(OUTDIR)/output.jld2" out ll lastState c metrics
@save "$(OUTDIR)/reduced_data/reduced_cb.jld2" deepcopy(cbData)

Cytof5.Model.logger("MCMC Completed.");
