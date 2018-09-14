import Dates

const monitor_default = Vector{Vector{Symbol}}([])
const thin_default = Vector{Int}()

function gibbs(init,
               update::Function;
               monitors::Vector{Vector{Symbol}}=deepcopy(monitor_default),
               thins::Vector{Int}=deepcopy(thin_default),
               nmcmc::Int64=1000, nburn::Int=0, printProgress::Bool=true)
  """
  This is my docs...
  """
  state = deepcopy(init)

  # Checking number of monitors.
  numMonitors = length(monitors)
  println("Number of monitors: $(numMonitors)")
  @assert numMonitors == length(thins)

  # Check monitor
  if numMonitors == 0
    println("Using default monitor.")
    fnames = [ fname for fname in fieldnames(typeof(init))]
    append!(monitors, [fnames])
    append!(thins, 1)
    numMonitors = 1
  end

  # object to return
  #out = Vector{Vector{Dict{Symbol, Any}}}([]) # monitors, chain, dict
  #start=Vector{Dict{Symbol, Any}}([])
  out = [ Vector{Dict{Symbol, Any}}([]) for i in 1:numMonitors ]

  # Milestones
  milestone = Int((nburn + nmcmc) / 10)

  function printMsg(i::Int)
    if i % milestone == 0 && printProgress
      println("$(Dates.now()) -- $i / $(nburn+nmcmc)")
    end
  end

  # burn in
  for i in 1:nburn
    printMsg(i)
    update(state, i, out)
  end


  # Gibbs loop
  for i in 1:nmcmc
    printMsg(i + nburn)
    update(state, i, out)

    for j in 1:numMonitors
      if i % thins[j] == 0
        substate = Dict{Symbol, Any}()
        for s in monitors[j]
          substate[s] =  deepcopy(getfield(state, s))
        end
        append!(out[j], [substate])
      end
    end
  end

  return (out, state)
end
