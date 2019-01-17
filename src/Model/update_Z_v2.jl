function update_Z_v2(s::State, c::Constants, d::Data, tuners::Tuners)
  # if 0.5 > rand()
  if true
    # update Z marginalizing over lam and gam
    update_Z_marg_lamgam(s, c, d)
  else
    # update Z jointly per marker, marginalizing over lam and gam
    update_Z_byrow(s, c, d, tuners)
  end
end

function update_Z_marg_lamgam(j::Int, k::Int, s::State, c::Constants, d::Data)
  Z0 = deepcopy(s.Z)
  Z0[j, k] = false 
  lp0 = log(1 - s.v[k]) + log_dmix_nolamgam(Z0, s, c, d)

  Z1 = deepcopy(s.Z)
  Z1[j, k] = true 
  lp1 = log(s.v[k]) + log_dmix_nolamgam(Z1, s, c, d)

  p_post = 1 / (1 + exp(lp0 - lp1))
  new_Zjk = p_post > rand()
  if new_Zjk != s.Z[j, k]
    println("Z marginal update was useful!")
  end

  s.Z[j, k] = new_Zjk
end

function update_Z_marg_lamgam(s::State, c::Constants, d::Data)
  for j in 1:d.J
    for k in 1:c.K
      update_Z_marg_lamgam(j, k, s, c, d)
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


