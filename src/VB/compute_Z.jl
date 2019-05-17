"""
This enables the backward gradient computations via Z.
Notice at the end, we basically return Z (binary tensor).
But we make use of the smoothed Z, which is differentiable.
We detach (Z - smoothed_Z) so that the gradients are not 
computed, and then add back smoothed_Z for the return.
The only gradient will then be that of smoothed Z.
"""
function compute_Z(v::AbstractArray, H::AbstractArray;
                   use_stickbreak::Bool=false, tau::Float64=.001)
  v_rs = reshape(v, 1, length(v))
  p = use_stickbreak ? cumprod(v_rs, dims=2) : v_rs
  r = ADVI.logit_safe.(p) .- ADVI.logit_safe.(H)
  smoothed_Z = ADVI.sigmoid_safe.(r / tau)
  Z = (p .> H)
  return Tracker.data(Z - smoothed_Z) + smoothed_Z
end

