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
                    printProgress::Bool=true, numPrints::Int=10,
                    flushOutput::Bool=false)

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

  loglike = Vector{Float64}()

  function update(s::State, i::Int, out)
    update_state(s, c, d, tuners, loglike)
  end

  out, lastState = MCMC.gibbs(init, update, monitors=monitors,
                              thins=thins, nmcmc=nmcmc, nburn=nburn,
                              printProgress=printProgress, numPrints=numPrints,
                              loglike=loglike, flushOutput=flushOutput)

  return out, lastState, loglike
end

end # Model
