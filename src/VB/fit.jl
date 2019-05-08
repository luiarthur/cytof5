function fit(; y::Vector{M}, niters::Int, batchsize::Int, c::Constants,
             opt=ADAM(1e-2), print_freq::Int=10, save_freq::Int=50,
             init::Union{StateMP, Nothing}=nothing,
             seed=nothing, flushOutput::Bool=false,
             return_init::Bool=false) where M <: Matrix

  # Set a random seed for reproducibility if seed is provided
  if seed != nothing
    Random.seed!(seed)
  end

  # Create output dictionary
  d = Dict{Symbol, Any}()

  # Initialize if needed
  if init == nothing
    init = State(c)
  end

  # Create metrics dictionary
  metrics = Dict(m => Float64[] for m in (:ll, :lp, :lq, :elbo))

  # Make a deep copy of initial state
  state = deepcopy(init)

  # Collect all model parameters and variational parameters
  ps = ADVI.vparams(state)

  # Create loss function
  loss(y) = -compute_elbo(state, y, c, metrics) / sum(c.N)

  # Create an object to store history of state
  state_hist = typeof(state)[]
  
  # Main loop
  @time for iter in 1:niters
    # Sample minibatch indices
    idx = [if 0 < batchsize < c.N[i] 
             Distributions.sample(1:c.N[i], batchsize, replace=false)
           else
             1:c.N[i]
           end for i in 1:c.I]

    # minibatch
    y_mini = [y[i][idx[i], :] for i in 1:c.I]

    # Compute and update gradients. Two different ways.
    # Flux.train!(loss, ps, [(y_mini, )], opt)
    gs = Tracker.gradient(() -> loss(y_mini), ps)
    Flux.Tracker.update!(opt, ps, gs)

    if iter % print_freq == 0
      m = ["$(key): $(round(metrics[key][end] / sum(c.N), digits=3))"
           for key in keys(metrics)]
      println("$(ShowTime()) | $(iter)/$(niters) | $(join(m, " | "))")
      append!(state_hist, [deepcopy(state)])
    end

    if flushOutput
      flush(stdout)
    end
  end

  # Put things in output
  out[:metrics] = metrics
  out[:state_hist] = state_hist
  out[:state] = state
  if return_init
    out[:init] = init
  end
  out[:batchsize] = batchsize
  out[:niters] = niters
  out[:opt] = opt
  out[:seed] = seed
  out[:c] = c

  return out
end
