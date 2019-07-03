using Random
using Distributions
using BSON
using DelimitedFiles

using RCall
@rimport base
@rimport FlowSOM
@rimport flowCore
@rimport mclust

# Path to scratch
RESULTS_DIR = "/scratchdata/alui2/cytof/results/"

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
  out = flowCore::flowFrame(Y)
  Y = NULL
  out 
  """
end

### Main ###
function main(data_path)
  # Read data
  simdat = BSON.load(data_path)
  y = simdat[:simdat][:y]
  y = [replaceMissing(yi, -6.0) for yi in y]
  I = length(y)
  N = [size(yi, 1) for yi in y]
  Y = vcat(y...)
  J = size(Y, 2)

  # outpath = "results/cb/flowsom"
  # mkpath(outpath)
  for i in 1:I
    # ff_Y = flowsanitize(y[i])
    ff_Y = flowsanitize(y[i])
    println("Running FlowSOM ...")
    @time fsom = FlowSOM.FlowSOM(ff_Y,
                                 colsToUse=1:J,  # columns to use
                                 maxMeta=20,  # Meta clustering option
                                 seed=42)  # Seed for reproducible results:

    fsmeta = fsom[:metaclustering]
    fsclus = fsmeta[Int.(convert(Matrix, fsom[:FlowSOM][:map][:mapping])[:, 1])]

    # print output
    ari = mclust.adjustedRandIndex(fsclus, simdat[:simdat][:lam][i])
    println("ari $(i): $ari")

    # writedlm("$(outpath)/clustering_$(i).txt", clus_i)
  end
end

# data path
SMALL_DATA_PATH = "$(RESULTS_DIR)/sim/small/best/output.bson"
BIG_DATA_PATH = "$(RESULTS_DIR)/sim/big/best/output.bson"

main(SMALL_DATA_PATH)
main(BIG_DATA_PATH)
