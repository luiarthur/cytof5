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

function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int64=1000, nburn::Int=1000, 
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

  # DIC
  if computeDIC
    # TODO
    #dtype = typeof(d.y)
    #dicStream = MCMC.DICstream{dtype}(d.y)
  end

  
  function update(s::State, iter::Int, out)
    update_state(s, c, d, tuners, loglike)

    # Inverse likelihood for each data point
    invLike = [ [ 1 / compute_like(i, n, j, s, c, d) for n in 1:d.N[i], j in 1:d.J ]
                 for i in 1:d.I ]

    if computeLPML
      # Update COP
      MCMC.updateCPO(cpoStream, invLike)

      # Add to printMsg
      milestone = floor((nburn + nmcmc) / numPrints)
      if milestone > 0 && iter % milestone == 0 && printProgress
        println(" -- LPML: $(MCMC.computeLPML(cpoStream))")
      end
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
    for (k, v) in metrics
      println("$k => $v")
    end
    return out, lastState, loglike, metrics
  else
    return out, lastState, loglike
  end
end

end # Model
