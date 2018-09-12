# TODO: Test
function update_alpha(s::State, c::Constants, d::Data)
  newShape = shape(c.alpha) + c.K
  newRate = rate(c.alpha) - sum(log.(s.v)) / c.K

  post = Gamma(newShape, 1.0 / newRate)

  s.alpha = rand(post)
end
