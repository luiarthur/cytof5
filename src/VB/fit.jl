function fit(; y::Vector{M}, niters::Int, batchsize::Int, c::Constants,
             opt=ADAM(1e-2), print_freq::Int=10, nsave::Int=0,
             init::Union{StateMP, Nothing}=nothing,
             seed=nothing, flushOutput::Bool=false,
             verbose::Int=1) where M <: Matrix

  if verbose >= 1
    println("opt: $opt")
    println("niters: $niters")
    println("batchsize: $batchsize")
    printConstants(c)
  end

  # Set a random seed for reproducibility if seed is provided
  if seed != nothing
    Random.seed!(seed)
  end

  # Set save frequency
  if 0 < nsave < niters
    save_freq = ceil(Int, niters / nsave)
  elseif nsave >= niters
    save_freq = 1
  else # nsave <= 1
    save_freq = niters 
  end

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

  # Print message
  function printMsg(iter::Integer)
    if iter % print_freq == 0
      m = ["$(key): $(round(metrics[key][end] / sum(c.N), digits=3))"
           for key in keys(metrics)]
      println("$(ShowTime()) | $(iter)/$(niters) | $(join(m, " | "))")
    end
  end

  # Create loss function
  function loss(y_batch::T) where T
    out = -compute_elbo(state, y_batch, c, metrics) / sum(c.N)
    if isnan(out) || isinf(out)
      out = zero(out)
      printMsg(0)
      println("Skipping update -- Loss was NaN or Inf")
    end
    return out
  end

  # Create an object to store history of state
  state_hist = [deepcopy(init)]
  
  # Main loop
  println("Compiling model...")
  @time for iter in 1:niters
    # Sample minibatch
    y_mini = [if 0 < batchsize < c.N[i]
                idx = Distributions.sample(1:c.N[i], batchsize, replace=false)
                y[i][idx, :]
              else
                y[i]
              end for i in 1:c.I]

    # Compute and update gradients. Two different ways.
    # Flux.train!(loss, ps, [(y_mini, )], opt)
    gs = Tracker.gradient(() -> loss(y_mini), ps)
    Flux.Tracker.update!(opt, ps, gs)

    printMsg(iter)

    if iter % save_freq == 0
      append!(state_hist, [deepcopy(state)])
    end
    # println("eps: $(rsample(state)[2].eps)")

    if flushOutput
      flush(stdout)
    end
  end

  # Put things in output
  out = Dict(:metrics => metrics,
             :state_hist => state_hist,
             :state => state,
             :batchsize => batchsize,
             :niters => niters,
             :seed => seed,
             :c => c)

  return out
end
