function update_v(k::Int, s::State, c::Constants, d::Data)
  sumZk = sum(s.Z[:, k])
  newA = s.alpha / c.K + sumZk
  newB = 1 + d.J - sumZk

  s.v[k] = rand(Beta(newA, newB))
end

function update_v_sb(k::Int, s::State, c::Constants, d::Data, tuners::Tuners)
  lp(vk::Float64) = log(s.alpha) + (s.alpha - 1.0) * log(vk) 

  function ll(vk::Float64)
    new_v = deepcopy(s.v)
    new_v[k] = vk
    b = cumprod(new_v)

    out = 0.0
    for j in 1:d.J
      for q in k:c.K
        if s.Z[j, q]
          out += log(b[q])
        else
          out += log1p(-b[q])
        end
      end
    end 

    return out
  end

  s.v[k] = MCMC.metLogitAdaptive(s.v[k], ll, lp, tuners.v[k])
end

function update_v(s::State, c::Constants, d::Data, tuners::Tuners, sb_ibp::Bool)
  for k in 1:c.K
    if sb_ibp
      update_v_sb(k, s, c, d, tuners)
    else
      update_v(k, s, c, d)
    end
  end
end
