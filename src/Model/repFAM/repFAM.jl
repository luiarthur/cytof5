"""
solve for the parameters in the similarity function in the rep-FAM model
"""
function solve_rep_param(; d=[1.0, 10.0], p=[.01, .5])
  alpha = log(log(1-p[1]) / log(1-p[2])) / log(abs(d[1] / d[2]))
  phi = - abs(d[1]) ^ alpha / log(1-p[1])
  return (alpha=alpha, phi=phi)
end


#= Test
using Distributions
using RCall
include("repFAM.jl")
d = [7, 10]
p = [.01, .99]
alpha, phi = solve_rep_param(d=d, p=p)
@assert abs(1-exp(-d[2]^alpha/phi) - p[2]) < 1E-6

R"d = 0:32; plot(d, 1-exp(-d^$alpha/$phi), type='o', pch=20); abline(v=$d[2], h=$p[2], lty=2)"
=#


"""
Simulate random bit vectors

For a bit vector of length K, estimate the probability of any two random bit vectors having 
at most k ∈ 1:K repeats.
"""
function simulate_prob_differing_bits(bit_vec_length::Int; nsims=100)
  base_bitvec = fill(false, bit_vec_length)
  sim_vec() = [rand() > .5 for j in 1:bit_vec_length]
  num_diffs() = sum(sim_vec() .!= base_bitvec)
  est_prob_num_diffs(k_repeats::Int) = mean(num_diffs() <= k_repeats for i in 1:nsims)

  return [est_prob_num_diffs(k) for k in 1:bit_vec_length]
end


"""
Find the tolerable number of repeated elements in a bit vec of length `bit_vec_length`,
based on a maximum probability of repeats being `thresh_prob`.
"""
function tolerable_num_diffs(bit_vec_length::Int; thresh_prob::Float64=.01, nsims::Int=100)
  sims = simulate_prob_differing_bits(bit_vec_length, nsims=nsims)
  return minimum(findall(sims .> thresh_prob))
end


"""
default parameters psi=(alpha, phi) for repFAM similarity function
"""
function default_psi(bit_vec_length::Int; thresh_probs=(.01, .5), nsims::Int=100)
  tol_num_diff_bits = tolerable_num_diffs(bit_vec_length, thresh_prob=thresh_probs[1], nsims=nsims)
  return solve_rep_param(d=[tol_num_diff_bits, tol_num_diff_bits+1], p=thresh_probs)
end


"""
return default similarity function for repFAM
"""
function gen_similarity_fn(bit_vec_length::Int, thresh_probs=(.01, .5), nsims::Int=100)
  psi = default_psi(bit_vec_length, thresh_probs=thresh_probs, nsims=nsims)
  return (z1::Vector{Bool}, z2::Vector{Bool}) -> exp(-sum(abs.(z1 - z2)) ^ psi.alpha / psi.phi)
end


"""
return default similarity function for repFAM, given probability of repeats
"""
function gen_similarity_fn(; repeats=(1,3), thresh_probs=(.01, .8))
  psi = solve_rep_param(d=repeats, p=thresh_probs)
  return (z1::Vector{Bool}, z2::Vector{Bool}) -> exp(-sum(abs.(z1 - z2)) ^ psi.alpha / psi.phi)
end


"""
Generate a function of threshold, which returns similarity of one if distance
between two vectors < threshold, and zero otherwise.
"""
function sim_fn_abs(threshold::Int)::Function
  return (z1::Vector{Bool}, z2::Vector{Bool}) -> sum(abs.(z1 - z2)) <= threshold ? 1.0 : 0.0
end


#= Test
using RCall, Distributions
include("repFAM.jl")
@time x = simulate_prob_differing_bits(25, nsims=1000)
R"plot(1:length($x), $x, xlab='k = number of diffs', ylab='probability of at most k differences', type='o')"
R"abline(h=.01)"

tolerable_num_diffs(25, thresh_prob=.01, nsims=1000)
=#

#=
using RCall, Distributions

J = 32
sim_fn = gen_similarity_fn()
sim_fn = sim_fn_abs(2)
z1 = [rand() > .5 ? 1 : 0 for i in 1:J] 
z2 = z1 .+ 0

p = []
for j in 1:J
  z2[j] = z2[j] == 1 ? 0 : 1
  append!(p, sim_fn(z1, z2))
end

R"plot(1:$J, $(1 .- p), type='o', xlab='differences', ylab='prior prob')"
=#

include("update_Z_repFAM.jl")
