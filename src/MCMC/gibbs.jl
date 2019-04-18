import Dates

const monitor_default = Vector{Vector{Symbol}}([])
const thin_default = Int[]

function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  for field in fields
    substate[field] = deepcopy(getfield(state, field))
  end

  return substate
end

showtime() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

"""
TODO...
"""
function gibbs(init::T,
               update::Function;
               monitors::Vector{Vector{Symbol}}=deepcopy(monitor_default),
               thins::Vector{Int}=deepcopy(thin_default),
               nmcmc::Int64=1000, nburn::Int=0,
               printFreq::Int=0, loglike=missing,
               flushOutput::Bool=false, printlnAfterMsg::Bool=true) where T

  @assert printFreq >= -1
  if printFreq == 0
    numPrints = 10
    printFreq = Int(ceil((nburn + nmcmc) / numPrints))
  end


  state = deepcopy(init)

  # Checking number of monitors.
  numMonitors = length(monitors)
  println("Number of monitors: $(numMonitors)")
  @assert numMonitors == length(thins)

  # Check monitor
  if numMonitors == 0
    println("Using default monitor.")
    fnames = [ fname for fname in fieldnames(typeof(init)) ]
    append!(monitors, [fnames])
    append!(thins, 1)
    numMonitors = 1
  end

  # Number of Samples for each Monitor
  numSamps = [ div(nmcmc, thins[i]) for i in 1:numMonitors ]
  #out = [ Vector{Dict{Symbol, Any}}([]) for i in 1:numMonitors ]

  if printFreq > 0
    println("Preallocating memory...")
  end
  # Object to return
  @time out = [ fill(deepcopyFields(state, monitors[i]), numSamps[i]) for i in 1:numMonitors ]

  function printMsg(i::Int)
    if printFreq > 0 && i % printFreq == 0
      loglikeMsg = ismissing(loglike) ? "" : "-- loglike: $(last(loglike))"
      print("$(showtime()) -- $(i)/$(nburn+nmcmc) $loglikeMsg")

      if printlnAfterMsg
        println()
      end

      if flushOutput
        flush(stdout)
      end
    end
  end

  # burn in
  for i in 1:nburn
    printMsg(i)
    update(state, i, out)
  end


  counters = zeros(Int, numMonitors)
  # Gibbs loop
  for i in 1:nmcmc
    printMsg(i + nburn)
    update(state, i + nburn, out)

    for j in 1:numMonitors
      if i % thins[j] == 0
        #substate = Dict{Symbol, Any}()
        #for s in monitors[j]
        #  substate[s] =  deepcopy(getfield(state, s))
        #end
        substate = deepcopyFields(state, monitors[j])
        #append!(out[j], [substate])
        counters[j] += 1
        out[j][counters[j]] = substate
      end
    end
  end

  return (out, state)
end
