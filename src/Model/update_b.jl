function update_b0(i::Int, s::State, c::Constants, d::Data)
  function ll(b0i::Float64)
  end

  function lp(b0i::Float64)
  end

  #return MCMC.metropolisAdaptive()
end

function update_b1(i::Int, s::State, c::Constants, d::Data)
end

function update_b(s::State, c::Constants, d::Data)
  for i in 1:d.I
    update_b0(i, s, c, d)
    update_b1(i, s, c, d)
  end
end
