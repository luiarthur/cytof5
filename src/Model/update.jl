include("update_Z.jl")
include("update_mus.jl")
include("update_alpha.jl")
include("update_v.jl")
include("update_W.jl")
include("update_sig2.jl")
include("update_eta.jl")
include("update_lam.jl")
include("update_gam.jl")
include("update_y_imputed.jl")
include("update_b.jl")

function update(s::State, c::Constants, d::Data)
  update_Z(s, c, d)
  update_mus(s, c, d)
  update_alpha(s, c, d)
  update_v(s, c, d)
  update_W(s, c, d)
  update_sig2(s, c, d)
  update_eta(s, c, d)
  update_lam(s, c, d)
  update_gam(s, c, d)

  update_y_imputed(s, c, d)
  update_b(s, c, d)
end