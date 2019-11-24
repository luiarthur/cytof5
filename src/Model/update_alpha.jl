# TODO: Test
function update_alpha!(s::State, c::Constants, d::Data, sb_ibp::Bool)
  newShape = shape(c.alpha_prior) + c.K

  if sb_ibp
    newRate = rate(c.alpha_prior) - sum(log.(s.v))
  else
    newRate = rate(c.alpha_prior) - sum(log.(s.v)) / c.K
  end

  # NOTE: For debugging
  # println("In update_alpha! -- newShape: $(newShape), newRate: $(newRate)")
  # println("                    v: $(s.v)")
  post = Gamma(newShape, 1.0 / newRate)

  s.alpha = rand(post)
end
