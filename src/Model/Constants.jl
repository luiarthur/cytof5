# TODO: Test
struct Constants
  alpha_prior::Tuple{Float16, Float16} # alpha ~ IG(shape, scale)
  mus_prior::Dict{Tuple{Int8, Int8}, Tuple{Float16, Float16}} # mu*[z,l] ~ TN(mean,sd)
  W_prior::Vector{Float16} # W_i ~ Dir_K(d)
  sig2_prior::Tuple{Float16, Float16} # sig2_i ~ IG(shape, scale)
  b0_prior::Tuple{Float16, Float16} # b0 ~ Normal(mean, sd)
  b1_prior::Tuple{Float16, Float16} # b1 ~ Gamma(shape, rate) 
end
