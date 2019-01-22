function update_Z_v2(s::State, c::Constants, d::Data, tuners::Tuners)
  # if 0.5 > rand()
  if 0.1 > rand()
    # update Z marginalizing over lam and gam
    update_Z_marg_lamgam(s, c, d)
  else
    # update Z jointly per marker, marginalizing over lam and gam
    # update_Z_byrow(s, c, d, tuners)
    #
    # update Z v1
    update_Z(s::State, c::Constants, d::Data)
  end
end

function update_Z_marg_lamgam(j::Int, k::Int,
                              A::Vector{Vector{Float64}},
                              B0::Vector{Matrix{Float64}},
                              B1::Vector{Matrix{Float64}},
                              s::State, c::Constants, d::Data)
  Z0 = deepcopy(s.Z)
  Z0[j, k] = false 
  lp0 = log(1 - s.v[k]) + log_dmix_nolamgam(Z0, A, B0, B1, s, c, d)

  Z1 = deepcopy(s.Z)
  Z1[j, k] = true 
  lp1 = log(s.v[k]) + log_dmix_nolamgam(Z1, A, B0, B1, s, c, d)

  p1_post = 1 / (1 + exp(lp0 - lp1))
  new_Zjk_is_one = p1_post > rand()
  # if new_Zjk_is_one != s.Z[j, k]
  #   println("Z marginal update was useful!")
  # end

  s.Z[j, k] = new_Zjk_is_one
end

function update_Z_marg_lamgam(s::State, c::Constants, d::Data)
  # Precompute A, B0, B1
  A = [[logdnoisy(i, n, s, c, d) for n in 1:d.N[i]] for i in 1:d.I]
  B0 = [[logdmixture(0, i, n, j, s, c, d) for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]
  B1 = [[logdmixture(1, i, n, j, s, c, d) for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]


  for j in 1:d.J
    for k in 1:c.K
      update_Z_marg_lamgam(j, k, A, B0, B1, s, c, d)
    end
  end
end

function update_Z_byrow(s::State, c::Constants, d::Data, tuners::Tuners)
  for j in 1:d.J
    update_Z_byrow(j, s, c, d, tuners)
  end
end

function update_Z_byrow(j::Int, s::State, c::Constants, d::Data, tuners::Tuners)
  # TODO
end


