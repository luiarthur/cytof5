import Cytof5.VB, Cytof5.VB.ADVI

StateTA = typeof(VB.State(VB.TA{Float64}))
StateArray = typeof(VB.State(Array{Float64}))

function data(s::StateTA)
  out = VB.State(Array{Float64})
  for k in fieldnames(typeof(s))
    if !(k in (:y_m, :y_log_s))
      f = getfield(s, k)
      setfield!(out, k, f.data)
    end
  end
  return out
end

function relabel_lam(lami_est, wi_mean)
  K = length(wi_mean)
  k_ord = sortperm(wi_mean, rev=true)
  lami_new = lami_est .+ 0
  counts = Int[]
  for k in 1:K
    idx_k = (lami_est .== k_ord[k])
    lami_new[idx_k] .= k
    append!(counts, sum(idx_k))
  end
  idx_0 = (lami_est .== 0)
  lami_new[idx_0] .= K + 1
  append!(counts, sum(idx_0))

  return lami_new, k_ord
end

function compress_simdat!(simdat)
  simdat[:y] = Matrix{Float16}.(simdat[:y])
  simdat[:y_complete] = Matrix{Float16}.(simdat[:y_complete])
  simdat[:lam] = Vector{Int8}.(simdat[:lam])
  simdat[:gam] = Matrix{Int8}.(simdat[:gam])

  return simdat
end

function decompress_simdat!(simdat)
  simdat[:y] = Matrix{Float64}.(simdat[:y])
  simdat[:y_complete] = Matrix{Float64}.(simdat[:y_complete])
  simdat[:lam] = Vector{Int}.(simdat[:lam])
  simdat[:gam] = Matrix{Int}.(simdat[:gam])

  return simdat
end
