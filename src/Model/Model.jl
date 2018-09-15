module Model

using Distributions

import LinearAlgebra
import Random

include("../MCMC/MCMC.jl")
include("State.jl")
include("Data.jl")
include("Constants.jl")
include("update.jl")

function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int64=1000, nburn::Int=1000, 
                    monitors=deepcopy(MCMC.monitor_default),
                    thins::Vector{Int}=deepcopy(MCMC.thin_default),
                    printProgress::Bool=true)

  function update(s::State, i::Int, out)
    update_state(s, c, d)
  end

  out, lastState = MCMC.gibbs(init, update, monitors=monitors,
                              thins=thins, nmcmc=nmcmc, nburn=nburn,
                              printProgress=printProgress)
  return out, lastState
end

end # Model
