module HMC
using Flux

struct Linear
  W
  b
end

function Linear(; dims, bias::Bool=true)
  W = randn(dims)
  if bias
    b = randn(1, dims[2])
  else
    b = 0
  end
  Linear(param(W), param(b))
end

(m::Linear)(x) = x * m.W .+ m.b


function runtest(niters; num_obs=10000, lr=.01)
  # Simulate random predictors
  X = randn(num_obs, 1)

  # Set parameter truth
  W_true = 2
  b_true = 1

  # function to simulate response
  f(X, w, b) = w * X .+ b + randn(size(X, 1), 1) * .1

  # simulate response
  y = f(X, W_true, b_true)

  # numer of features
  num_features = size(X, 2)

  # build model
  model = HMC.Linear(dims=(num_features, 1))

  # loss function
  loss(x, y) = Flux.mean((model(x) - y) .^ 2)

  # parameters to track
  ps = Flux.Params([getfield(model, fname)
                    for fname in fieldnames(typeof(model))])

  # gradient descent optimizer
  opt = Descent(lr)

  # Training loop
  for i in 1:niters
    # NOTE: These versions are equivalent

    # Version I
    # Flux.train!(loss, ps, [(X, y)], opt)

    # Version II
    # gs = Tracker.gradient(() -> loss(X, y), ps)
    # Flux.Tracker.update!(opt, ps, gs)

    # Version III. Could be used in an HMC step.
    # See p.14 of https://arxiv.org/pdf/1206.1901.pdf
    gs = Tracker.gradient(() -> loss(X, y), ps)
    for p in ps
      # Update parameter `p` to have value `p += -gs[p]*lr`
      Flux.Tracker.update!(p, -gs[p] * lr)
    end
  end

  # Print result
  println("model: $(model) | truth: $((W_true, b_true)) | loss: $(loss(X, y))")
end

end  # end of module

#= Time the thing
@time HMC.runtest(100, num_obs=100, lr=.1)
=#
