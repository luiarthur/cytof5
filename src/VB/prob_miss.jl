function prob_miss(y::R, b0::Float64, b1::Float64, b2::Float64) where {R <: Real}
  return sigmoid(b0 + b1*y + b2*(y^2))
end

function prob_miss(y::A, b0::Float64, b1::Float64, b2::Float64) where {A <: AbstractArray}
  return sigmoid.(b0 .+ b1*y + b2*(y .^ 2))
  # NOTE: Unnecessary broadcasting slows down backprop!
  # Don't do the following:
  # return sigmoid.(b0 .+ b1 .* y + b2 .* y .^ 2)
end

# linear miss mech
function prob_miss(y::R, b0::Float64, b1::Float64) where {R <: Real}
  return sigmoid(b0 + b1*y)
end
function prob_miss(y::A, b0::Float64, b1::Float64) where {A <: AbstractArray}
  return sigmoid.(b0 .+ b1*y)
end


