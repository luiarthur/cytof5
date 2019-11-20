function dden_complete(ygrid, W, eta, Z, mus, sig2; i, j)
  K = size(W, 2)
  L0 = size(eta[0], 3)
  L1 = size(eta[1], 3)
  L = Dict(0 => L0, 1 => L1)

  dden = [begin
            si = sqrt(sig2[i])
            dd = 0.0
            for k in 1:K
              ddk = 0.0
              z = Z[j, k]
              for ell in 1:L[z]
                mu_zl = mus[z][ell]
                eta_zijl = eta[z][i, j, ell]
                ddk += eta_zijl * pdf(Normal(mu_zl, si), y)
              end
              dd += W[i, k] * ddk
            end
            dd
          end for y in ygrid]

  return dden
end


function dden_complete(ygrid, W, eta, Z, mus, sig2)
  I, J = size(eta[0])[1:end-1]

  dden = [dden_complete(ygrid, W, eta, Z, mus, sig2, i=i, j=j)
          for i in 1:I, j in 1:J]

  return dden
end
