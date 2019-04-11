println("Loading packages...")
@time begin
  using Cytof5
  using Random, Distributions
  # TODO: Get rid of this dep
  using RCall
  using BSON
  using ArgParse
  import Cytof5.Model.logger
  include("util.jl")
end
println("Done loading packages.")


# TODO: review
# ARG PARSING
function parse_cmd()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--simdat_path"
      arg_type = String
      required = true
    "--MCMC_ITER"
      arg_type = Int
      default = 1000
    "--BURN"
      arg_type = Int
      default = 10000
    "--K_MCMC"
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

SIMDAT_PATH = PARSED_ARGS["simdat_path"]

MCMC_ITER = PARSED_ARGS["MCMC_ITER"]
BURN = PARSED_ARGS["BURN"]

K_MCMC = PARSED_ARGS["K_MCMC"]
L0_MCMC = PARSED_ARGS["L0_MCMC"]
L1_MCMC = PARSED_ARGS["L1_MCMC"]
L_MCMC = Dict(0 => L0_MCMC, 1 => L1_MCMC)

EXP_NAME = PARSED_ARGS["EXP_NAME"]
SEED = PARSED_ARGS["SEED"]
RESULTS_DIR = PARSED_ARGS["RESULTS_DIR"]
printFreq = PARSED_ARGS["printFreq"]
# END OF ARG PARSING

# CREATE RESULTS DIR
OUTDIR = "$(RESULTS_DIR)/$(EXP_NAME)/"
mkpath(OUTDIR)

# Set random seed
Random.seed!(SEED);

logger("Load simulated data ...");
BSON.@load SIMDAT_PATH simdat
dat = Cytof5.Model.Data(simdat[:y])

logger("Generating priors ...");
@time c = Cytof5.Model.defaultConstants(dat, K_MCMC, L_MCMC,
                                        tau0=10.0, tau1=10.0,
                                        sig2_prior=InverseGamma(3.0, 2.0),
                                        alpha_prior=Gamma(0.1, 10.0),
                                        yQuantiles=[0.0, .25, .5], pBounds=[.05, .8, .05], # near
                                        similarity_Z=Cytof5.Model.sim_fn_abs(10000),
                                        probFlip_Z=2.0 / (dat.J * K_MCMC),
                                        noisyDist=Normal(0.0, 3.16))
Cytof5.Model.printConstants(c)
println("dat.I: $(dat.I)")
println("dat.J: $(dat.J)")
println("dat.N: $(dat.N)")


# Plot missing mechanism
logger("Plot missing mechanism")
util.plotPdf("$(OUTDIR)/prob_miss.pdf")
R"par(mfrow=c($(dat.I), 1))"
for i in 1:dat.I
  util.plotProbMiss(c.beta, i)
end
R"par(mfrow=c(1,1))"
util.devOff()


logger("Generating initial state ...");
# @time init = Cytof5.Model.genInitialState(c, dat)
logger("use smart init ...")
@time init = Cytof5.Model.smartInit(c, dat)

# Plot initial Z
util.plotPdf("$(OUTDIR)/Z_init.pdf")
addGridLines(J::Int, K::Int, col="grey") = util.abline(v=(1:K) .+ .5, h=(1:J) .+ .5, col=col)
util.myImage(init.Z, xlab="Features", ylab="Markers", addL=false, f=Z->addGridLines(dat.J, c.K))
util.devOff()


# Fit Model
nsamps_to_thin(nsamps::Int, nmcmc::Int) = max(1, div(nmcmc, nsamps))

@time out, lastState, ll, metrics, dden=
  Cytof5.Model.cytof5_fit(init, c, dat,
                          monitors=[[:Z, :lam, :W,
                                     :sig2, :delta,
                                     :alpha, :v,
                                     :eta, :eps],
                                    [:y_imputed, :gam]],
                          thins=[2, nsamps_to_thin(10, MCMC_ITER)],
                          nmcmc=MCMC_ITER, nburn=BURN,
                          computeLPML=true, computeDIC=true,
                          computedden=true, thin_dden=nsamps_to_thin(200, MCMC_ITER),
                          use_repulsive=false,
                          joint_update_Z=true,
                          printFreq=10, flushOutput=true)

logger("Saving Data ...");
println("length of dden: $(length(dden))")
# @save "$(OUTDIR)/output.jld2" out dat ll lastState c dat metrics
BSON.@save "$(OUTDIR)/output.bson" out ll lastState c metrics init dden simdat

logger("MCMC Completed.");

#= Test
julia --color=yes sim.jl --I=3 --J=32 --N_factor=100 --K=8 --K_MCMC=10 --L0_MCMC=5 --L1_MCMC=5 \
      --MCMC_ITER=10 --BURN=10 --RESULTS_DIR=results --EXP_NAME=bla
=#
