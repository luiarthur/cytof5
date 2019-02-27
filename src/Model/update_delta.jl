function mus(z::Bool, s::State, c::Constants, d::Data)
  cs_delta_z = cumsum(s.delta[z])
  return z ? cs_delta_z : -cs_delta_z
end

function mus(z::Bool, l::Integer, s::State, c::Constants, d::Data)
  ds_zl = sum(s.delta[z][1:l])
  return z ? ds_zl : -ds_zl
end

function mus(i::Integer, n::Integer, j::Integer, s::State, c::Constants, d::Data)
  k = s.lam[i][n]

  # mean of the noisy class
  mus_inj = 0.0

  # look up mean
  if k > 0
    z = s.Z[j, k]
    l = s.gam[i][n, j]
    mus_inj = mus(z, l, s, c, d)
  end

  return mus_inj
end

function update_delta(s::State, c::Constants, d::Data)
  for z in 0:1
    for l in 1:c.L[z]
      update_delta(Bool(z), l, s, c, d)
    end
  end
end

# TODO: TEST!
function update_delta(z::Bool, l::Int, s::State, c::Constants, d::Data)
  (psi_z, tau_z, _, _) = params(c.delta_prior[z])

  cardinality = zeros(Int, d.I)
  g_sum = zeros(Float64, d.I)

  for i in 1:d.I
    for j in 1:d.J
      for n in 1:d.N[i]
        k = s.lam[i][n]
        if k > 0
          z_jk = s.Z[j, k]
          gam_inj = s.gam[i][n, j]
          if z_jk == z && gam_inj >= l
            cardinality[i] += 1

            dz = s.delta[z][1:gam_inj][1:end .!= l]
            g_sum[i] += (-1)^(1 - z) * s.y_imputed[i][n, j] - sum(dz)
          end
        end
      end
    end
  end

  denom = 1.0 + tau_z^2 * sum(cardinality ./ s.sig2[z][:, l])
  new_m = (psi_z + tau_z^2 * sum(g_sum ./ s.sig2[z][:, l])) / denom
  new_s = sqrt(tau_z^2 / denom)

  s.delta[z][l] = rand(TruncatedNormal(new_m, new_s, 0.0, Inf))
end
