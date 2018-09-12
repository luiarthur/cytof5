include("update_Z.jl")
include("update_mus.jl")
include("update_alpha.jl")

function update(s::State, c::Constants, d::Data)
  update_Z(s, c, d)
  update_mus(s, c, d)
  update_alpha(s, c, d)
end
