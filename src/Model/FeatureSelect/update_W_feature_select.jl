# Update W
function update_W(s::StateFS, c::ConstantsFS, d::DataFS)
  for i in 1:d.data.I
    update_W(i, s, c, d)
  end
end

function update_W(i::Int, s::StateFS, c::ConstantsFS, d::DataFS)
  wr_i = s.W_star[i, :] .* s.r[i, :]
  s.theta.W[i, :] .= wr_i ./ sum(wr_i, dims=2)
end

# Update W star
function update_W_star(s::StateFS, c::ConstantsFs, d::DataFS)
  for i in 1:d.data.I
    update_W_star(i, s, c, d)
  end
end

function update_W_star(i::Int, s::StateFS, c::ConstantsFS, d::DataFS)
  # TODO
end
