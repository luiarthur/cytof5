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
  # update y_imputed
  y_imputed::Dict{Tuple{Int,Int}, Float64}
  # update b0
  b0::Vector{Float64}
  # update b1
  b1::Vector{Float64}
  # update mus_inj
  mus::Dict{Tuple{Int,Int}, Float64}
  # update sig2_i
  sig2::Vector{Float64}
end

function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int=1000, nburn::Int=1000, 
                    monitors=[[:Z, :lam, :W, :b0, :b1, :v, :sig2, :mus,
                               :alpha, :v, :eta]],
                    thins::Vector{Int}=[1],
                    printProgress::Bool=true, numPrints::Int=10, flushOutput::Bool=false,
                    computeDIC::Bool=false, computeLPML::Bool=false)

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

  # CPO
  if computeLPML
    dtype = typeof(d.y)
    cpoStream = MCMC.CPOstream{dtype}(d.y)
  end

  # DIC. TODO
  if computeDIC
    function updateParams(d::MCMC.DICstream{DICparam}, param::DICparam)
      # update y_imputed
      # update b0
      # update b1
      # update mus_inj
      # update sig2_i
    end

    function paramMeanCompute(d::MCMC.DICstream{State})
    end

    loglike_fn(state::State) = compute_loglike(state, c, d, normalize=false)
    dicStream = MCMC.DICstream{State}(deepcopy(init), loglike_fn)
  end

  function printMsg(iter::Int, msg::String)
    milestone = floor((nburn + nmcmc) / numPrints)
    if milestone > 0 && iter % milestone == 0 && printProgress
      print(msg)
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
      printMsg(iter, " -- LPML: $(MCMC.computeLPML(cpoStream)) \n")
    else
      println()
    end

    if computeDIC
    end
  end

  out, lastState = MCMC.gibbs(init, update, monitors=monitors,
                              thins=thins, nmcmc=nmcmc, nburn=nburn,
                              printProgress=printProgress, numPrints=numPrints,
                              loglike=loglike, flushOutput=flushOutput,
                              printlnAfterMsg=!(computeDIC || computeLPML))

  if computeDIC || computeLPML
    LPML = computeLPML ? MCMC.computeLPML(cpoStream) : NaN
    DIC = computeDIC ? MCMC.computeDIC(DICstream) : NaN
    metrics = Dict(:LPML => LPML, :DIC => DIC)
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
