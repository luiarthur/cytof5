using Random
using Distributions
using JLD2, FileIO
using DelimitedFiles

using RCall
@rimport base
@rimport FlowSOM
@rimport flowCore

# Path to scratch
RESULTS_DIR = "/scratchdata/alui2/cytof/results/"

# CB data path
CB_PATH = "$(RESULTS_DIR)/cb/best/reduced_data/reduced_cb.jld2"

"""
read a single object from a jld2 file
"""
function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

"""
replace missing values (NaN) in yi with x
"""
function replaceMissing(yi, x)
  out = deepcopy(yi)
  out[isnan.(out)] .= x
  return out
end


function flowsanitize(Y)
  @rput Y
  return R"""
  colnames(Y) = 1:NCOL(Y)
  flowCore::flowFrame(Y)
  """
end

### Main ###
function main()
  # Read data
  y = loadSingleObj(CB_PATH)
  I = length(y)
  N = [size(yi, 1) for yi in y]
  Y = replaceMissing(vcat(y...), -6)
  J = size(Y, 2)
  ff_Y = flowsanitize(Y)

  println("Running FlowSOM ...")
  @time fsom = FlowSOM.FlowSOM(ff_Y,
                               colsToUse=1:J,  # columns to use
                               maxMeta=20,  # Meta clustering option
                               seed=42)  # Seed for reproducible results:

  idx_upper = cumsum(N) 
  idx_lower = [1; idx_upper[1:end-1] .+ 1]
  idx = [idx_lower idx_upper]

  fsmeta = fsom[:metaclustering]
  fsclus = fsmeta[Int.(convert(Matrix, fsom[:FlowSOM][:map][:mapping])[:, 1])]

  outpath = "results/cb/flowsom"
  mkpath(outpath)
  for i in 1:I
    # print output
    clus_i = fsclus[idx[i, 1]:idx[i, 2]]

    writedlm("$(outpath)/clustering_$(i).txt", clus_i)
  end
end

main()
