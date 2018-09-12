struct Constants
  alpha_prior::Tuple{Float16, Float16} # alpha ~ IG(shape, scale)
  mus_prior::Dict{Tuple{Int8, Int8}, Tuple{Float16, Float16}} # mu*[z,l] ~ TN(mean,sd)
  W_prior::Vector{Float16} # W_i ~ Dir_K(d)
  sig2_prior::Tuple{Float16, Float16} # sig2_i ~ IG(shape, scale)
  b0_prior::Tuple{Float16, Float16} # b0 ~ Normal(mean, sd)
  b1_prior::Tuple{Float16, Float16} # b1 ~ Unif(a, b) (positive)
  K::Int
  L::Int
end

function defaultConstants(data::Data, K::Int, L::Int)
  alpha_prior = (3.0, 2.0)
  mus_prior = Dict{Tuple{Int8, Int8}, Tuple{Float16, Float16}}()
  vec_y = vcat(vec.(data.y)...)
  y_neg = filter(y_inj -> !ismissing(y_inj) && y_inj < 0, vec_y)
  y_pos = filter(y_inj -> !ismissing(y_inj) && y_inj > 0, vec_y)
  for z in 0:1
    for l in 1:L
      if z == 0
        mus_prior[z, l] = (mean(y_neg), std(y_neg))
      else
        mus_prior[z, l] = (mean(y_pos), std(y_pos))
      end
    end
  end
  W_prior = fill(Float16(1.0/K), K)
  sig2_prior = (3.0, 2.0)
  b0_prior = (0.0, 10.0)
  b1_prior = (0.0, 50.0)

  Constants(alpha_prior, mus_prior, W_prior, sig2_prior, b0_prior, b1_prior, K, L)
end
