module Model

using Distributions
using RCall # Mclust
import LinearAlgebra # Identity matrix
import Random # shuffle
import StatsBase # wsample

include("../MCMC/MCMC.jl")
import .MCMC.Util.@namedargs

include("util.jl")
include("State.jl")
include("Data.jl")
include("Constants.jl")
include("Tuners.jl")
include("update.jl")
include("repFAM.jl")
include("genInitialState.jl")

@namedargs mutable struct DICparam
  p::Vector{Matrix{Float64}}
  mu::Vector{Matrix{Float64}}
  sig::Vector{Vector{Float64}}
end

"""
printFreq: defaults to 0 => prints every 10%. turn off printing by setting to -1.
"""
function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int=1000, nburn::Int=1000, 
                    monitors=[[:Z, :lam, :W, :v, :sig2, :mus, :alpha, :v, :eta]],
                    fix::Vector{Symbol}=Vector{Symbol}(),
                    thins::Vector{Int}=[1],
                    printFreq::Int=0, flushOutput::Bool=false,
                    computeDIC::Bool=false, computeLPML::Bool=false,
                    use_repulsive::Bool=false, verbose::Int=1)

  if verbose >= 1
    fixed_vars_str = join(fix, ", ")
    if fixed_vars_str == ""
      fixed_vars_str = "nothing"
    end
    println("fixing: $fixed_vars_str")
  end

  @assert printFreq >= -1
  if printFreq == 0
    numPrints = 10
    printFreq = Int(ceil((nburn + nmcmc) / numPrints))
  end


  y_tuner = begin
    dict = Dict{Tuple{Int, Int, Int}, MCMC.TuningParam}()
    for i in 1:d.I
      for n in 1:d.N[i]
        for j in 1:d.J
          if d.m[i][n, j] == 1
            dict[i, n, j] = MCMC.TuningParam(1.0)
          end
        end
      end
    end
    dict
  end
  tuners = Tuners(y_imputed=y_tuner, # yinj, for inj s.t. yinj is missing
                  Z=MCMC.TuningParam(MCMC.sigmoid(c.probFlip_Z, a=0.0, b=1.0)))

  # Loglike
  loglike = Vector{Float64}()

  function printMsg(iter::Int, msg::String)
    if printFreq > 0 && iter % printFreq == 0
      print(msg)
    end
  end

  # CPO
  if computeLPML
    invLike = [ [ 1.0 / compute_like(i, n, j, init, c, d) for n in 1:d.N[i], j in 1:d.J ]
                 for i in 1:d.I ]
    invLikeType = typeof(invLike)
    cpoStream = MCMC.CPOstream{invLikeType}(invLike)
  end

  # DIC
  if computeDIC
    local tmp = DICparam(p=deepcopy(d.y),
                         mu=deepcopy(d.y),
                         sig=[zeros(Float64, d.N[i]) for i in 1:d.I])
    dicStream = MCMC.DICstream{DICparam}(tmp)

    function updateParams(d::MCMC.DICstream{DICparam}, param::DICparam)
      d.paramSum.p += param.p
      d.paramSum.mu += param.mu
      d.paramSum.sig += param.sig
      return
    end

    function paramMeanCompute(d::MCMC.DICstream{DICparam})::DICparam
      return DICparam(d.paramSum.p ./ d.counter,
                      d.paramSum.mu ./ d.counter,
                      d.paramSum.sig ./ d.counter)
    end

    function loglikeDIC(param::DICparam)::Float64
      ll = 0.0

      for i in 1:d.I
        for j in 1:d.J
          for n in 1:d.N[i]
            y_inj_is_missing = (d.m[i][n, j] == 1)

            if y_inj_is_missing
              ll += logpdf(Bernoulli(param.p[i][n, j]), d.m[i][n, j])
            else
              ll += logpdf(Normal(param.mu[i][n, j], param.sig[i][n]), d.y[i][n, j])
            end
          end
        end
      end

      return ll
    end

    function convertStateToDicParam(s::State)::DICparam
      p = [[prob_miss(s.y_imputed[i][n, j], c.beta[:, i])
            for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]

      mu = [[s.lam[i][n] > 0 ? s.mus[s.Z[j, s.lam[i][n]]][s.gam[i][n, j]] : 0.0 
             for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]

      sig = [[s.lam[i][n] > 0 ? sqrt(s.sig2[i]) : sqrt(c.sig2_0) for n in 1:d.N[i]] for i in 1:d.I]

      return DICparam(p, mu, sig)
    end
  end


  function update(s::State, iter::Int, out)
    update_state(s, c, d, tuners, loglike, fix, use_repulsive)

    if computeLPML && iter > nburn
      # Inverse likelihood for each data point
      invLike = [ [ 1 / compute_like(i, n, j, s, c, d) for n in 1:d.N[i], j in 1:d.J ]
                   for i in 1:d.I ]

      # Update COP
      MCMC.updateCPO(cpoStream, invLike)

      # Add to printMsg
      printMsg(iter, " -- LPML: $(MCMC.computeLPML(cpoStream))")
    end

    if computeDIC && iter > nburn
      # Update DIC
      MCMC.updateDIC(dicStream, s, updateParams, loglikeDIC, convertStateToDicParam)

      # Add to printMsg
      printMsg(iter, " -- DIC: $(MCMC.computeDIC(dicStream, loglikeDIC,
                                                 paramMeanCompute))")
    end

    printMsg(iter, "\n")
    if flushOutput
      flush(stdout)
    end
  end

  if isinf(compute_loglike(init, c, d, normalize=true))
    println("Warning: Initial state yields likelihood of zero.")
    println("It is likely the case that the initialization of missing values is not consistent with the provided missing mechanism. The MCMC will almost certainly reject the initial values and sample new ones in its place.")
  else
    println("")
  end

  out, lastState = MCMC.gibbs(init, update, monitors=monitors,
                              thins=thins, nmcmc=nmcmc, nburn=nburn,
                              printFreq=printFreq,
                              loglike=loglike, flushOutput=flushOutput,
                              printlnAfterMsg=!(computeDIC || computeLPML))

  if computeDIC || computeLPML
    LPML = computeLPML ? MCMC.computeLPML(cpoStream) : NaN
    Dmean, pD = computeDIC ? MCMC.computeDIC(dicStream, loglikeDIC, paramMeanCompute,
                                             return_Dmean_pD=true) : (NaN, NaN)
    metrics = Dict(:LPML => LPML, :DIC => Dmean + pD, :Dmean => Dmean, :pD => pD)
    println()
    println("metrics:")
    for (k, v) in metrics
      println("$k => $v")
    end
    return out, lastState, loglike, metrics
  else
    return out, lastState, loglike
  end
end

#precompile(cytof5_fit, (State, Constants, Data, Int, Int, Vector{Vector{Symbol}}, Vector{Int}, Bool, Int, Bool, Bool, Bool))

end # Model
