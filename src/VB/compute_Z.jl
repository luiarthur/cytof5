"""
This enables the backward gradient computations via Z.
Notice at the end, we basically return Z (binary tensor).
But we make use of the smoothed Z, which is differentiable.
We detach (Z - smoothed_Z) so that the gradients are not 
computed, and then add back smoothed_Z for the return.
The only gradient will then be that of smoothed Z.
"""
function compute_Z(v::AbstractArray, H::AbstractArray;
                   use_stickbreak::Bool=false, tau::Float64=.005)
  v_rs = reshape(v, 1, length(v))
  p = use_stickbreak ? cumprod(v_rs, dims=2) : v_rs
  logit = p .- H
  smoothed_Z = sigmoid.(logit / tau)
  Z = (p .> H).data # becomes non-tracked
  return (Z - smoothed_Z.data) + smoothed_Z
end

