@namedargs mutable struct DICparam
  p::Vector{Matrix{Float64}}
  mu::Vector{Matrix{Float64}}
  sig::Vector{Vector{Float64}}
  y::Vector{Matrix{Float64}}
end
