module Model

using Distributions

import LinearAlgebra
import Random

include("../MCMC/MCMC.jl")
include("State.jl")
include("Data.jl")
include("Constants.jl")
include("Tuners.jl")
include("update.jl")

function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int64=1000, nburn::Int=1000, 
                    monitors=[[:Z, :lam, :W, :b0, :b1, :v, :sig2, :mus]],
                    thins::Vector{Int}=[1],
                    printProgress::Bool=true)

  y_tuner = begin
    dict = Dict{Tuple{Int, Int, Int}, MCMC.TuningParam}()
    for i in 1:d.I
      for n in 1:d.N[i]
        for j in 1:d.J
          if ismissing(d.y[i][n, j])
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

  function update(s::State, i::Int, out)
    update_state(s, c, d, tuners)
  end

  out, lastState = MCMC.gibbs(init, update, monitors=monitors,
                              thins=thins, nmcmc=nmcmc, nburn=nburn,
                              printProgress=printProgress)
  return out, lastState
end

end # Model
