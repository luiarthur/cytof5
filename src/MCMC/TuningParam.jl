mutable struct TuningParam{T}
  value::T
  acceptanceCount::Int
  currentIter::Int
  batch_size::Int
  delta::Function
  targetAcc::Float64

  TuningParam(v::V) where V = new{V}(v, 0, 0, 50, n::Int->min(n^(-0.5), 0.01), 0.44)
end

function update_tuning_param_default(param::TuningParam, accept::Bool)
  if accept
    param.acceptanceCount += 1
  end

  param.currentIter += 1

  if param.currentIter % param.batch_size == 0
    n = Int(floor(param.currentIter / param.batch_size))
    factor = exp(param.delta(n))
    if acceptanceRate(param) > param.targetAcc
      param.value *= factor
    else
      param.value /= factor
    end

    param.acceptanceCount = 0
  end

  return
end

function acceptanceRate(param::TuningParam)
  return param.acceptanceCount / param.batch_size
end

