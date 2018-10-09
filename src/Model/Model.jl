module Model

using Distributions

import LinearAlgebra
import Random # shuffle

include("../MCMC/MCMC.jl")
include("util.jl")
include("State.jl")
include("Data.jl")
include("Constants.jl")
include("Tuners.jl")
include("update.jl")

mutable struct DICparam
  p::Vector{Matrix{Float64}}
  mu::Vector{Matrix{Float64}}
  sig::Vector{Float64}
end

"""
printFreq: defaults to 0 => prints every 10%. turn off printing by setting to -1.
"""
function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int=1000, nburn::Int=1000, 
                    monitors=[[:Z, :lam, :W, :b0, :b1, :v, :sig2, :mus,
                               :alpha, :v, :eta]],
                    thins::Vector{Int}=[1],
                    printFreq::Int=0, flushOutput::Bool=false,
                    computeDIC::Bool=false, computeLPML::Bool=false)

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
  tuners = Tuners([MCMC.TuningParam(1.0) for i in 1:d.I], # b0i, 1...I
                  [MCMC.TuningParam(1.0) for i in 1:d.I], # b1i, 1...I
                  y_tuner) # yinj, for inj s.t. yinj is missing

  # Loglike
  loglike = Vector{Float64}()

  function printMsg(iter::Int, msg::String)
    if printFreq > 0 && iter % printFreq == 0
      print(msg)
    end
  end

  # CPO
  if computeLPML
    invLike = [ [ 1 / compute_like(i, n, j, init, c, d) for n in 1:d.N[i], j in 1:d.J ]
                 for i in 1:d.I ]
    invLikeType = typeof(invLike)
    cpoStream = MCMC.CPOstream{invLikeType}(invLike)
  end

  # DIC
  if computeDIC
    local tmp = DICparam(deepcopy(d.y), deepcopy(d.y), zeros(Float64, d.I))
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
                      d.paramSum.sig / d.counter)
    end

    function loglikeDIC(param::DICparam)::Float64
      ll = 0.0

      for i in 1:d.I
        for j in 1:d.J
          for n in 1:d.N[i]
            ll += logpdf(Bernoulli(param.p[i][n, j]), d.m[i][n, j])
            if d.m[i][n, j] == 0 # observed
              ll += logpdf(Normal(param.mu[i][n, j], param.sig[i]), d.y[i][n, j])
            end
          end
        end
      end

      return ll
    end

    function convertStateToDicParam(s::State)::DICparam
      p = [[prob_miss(s.y_imputed[i][n, j], s.b0[i], s.b1[i]) for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]
      mu = [[s.mus[s.Z[j, s.lam[i][n]]][s.gam[i][n, j]] for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]
      sig = sqrt.(s.sig2)
      return DICparam(p, mu, sig)
    end
  end


  function update(s::State, iter::Int, out)
    update_state(s, c, d, tuners, loglike)

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
