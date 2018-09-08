mutable struct TuningParam
  value::Any
  acceptanceCount::Int
  currentIter::Int
  batch_size::Int

  TuningParam(v) = new(v, 0, 0, 50)
end

function update(param::TuningParam, accept::Bool)

  if accept
    param.acceptanceCount += 1
  end

  param.currentIter += 1

  return
end

function acceptanceRate(param::TuningParam)
  #return param.acceptanceCount / param.currentIter
  return param.acceptanceCount / param.batch_size
end

