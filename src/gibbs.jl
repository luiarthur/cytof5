struct StateGeneric

end

const monitor_default = Vector{Vector{Symbol}}([])
const thin_default = Vector{Int}()

function gibbs(init,
               update::Function;
               monitors::Vector{Vector{Symbol}}=monitor_default,
               thins::Vector{Int}=thin_default,
               nmcmc::Int64=1000, nburn::Int=0, printProgress::Bool=true)
  """
  This is my docs...
  """

  # Checking number of monitors.
  numMonitors = length(monitors)
  println("Number of monitors: $(numMonitors)")
  @assert numMonitors == length(thins)

  # Check monitor
  if monitors === monitor_default
    println("Using default monitor.")
    fnames = [ fname for fname in fieldnames(typeof(init))]
    append!(monitors, [fnames])
    append!(thins, 1)
  end

  # burn in
  for i in 1:nburn

  end

  # Gibbs loop

  return init
end

struct State
  x::Int
  y::Float64
end

function update(s::State)::Void
  s.x += 1
  s.y -= 1
end

s = State(0, 0)
gibbs(s, update)


