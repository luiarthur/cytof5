function update_Z(s::State, c::Constants, d::Data)
  J = d.J
  K = c.K
  for j in 1:J
    for k in 1:K
      update_Zjk(s, c, d)
    end
  end
end


function update_Zjk(s::State, c::Constants, d::Data, j::Int, k::Int)
end
