import Dates

const monitor_default = Vector{Vector{Symbol}}([])
const thin_default = Int[]

"""
Partitions a list `xs` into `(good, bad)` according
to a condition.
"""
function partition(condition::Function, xs::Vector{T}) where T
  good = T[]
  bad = T[]

  for x in xs
    if condition(x)
      append!(good, [x])
    else
      append!(bad, [x])
    end
  end

  return good, bad
end

"""
Checks if a field is a subtype.
If a field has the form "a__b", the name
of the field is `a`, and it has a field `b`.
"""
issubtype(x::Symbol)::Bool = occursin("__", String(x))

function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  # Partition fields into subtypes and regular fields
  subtypefields, regfields = partition(issubtype, fields)

  # Get level1 fields
  for field in regfields
    substate[field] = deepcopy(getfield(state, field))
  end

  # Get level2 fields
  for field in subtypefields
    statename, _field = split(String(field), "__")
    _state = getfield(state, Symbol(statename))
    substate[field] = deepcopy(getfield(_state, Symbol(_field)))
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

  # Create object to return
  @time out = [fill(deepcopyFields(state, monitors[i]), numSamps[i])
               for i in 1:numMonitors]

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
