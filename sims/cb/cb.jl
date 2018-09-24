import Pkg
Pkg.activate("../../")

using Cytof5, Random, RCall
using JLD2, FileIO

Random.seed!(10)
printDebug = false

# Read CB Data
cbDataPath = "/soe/alui2/repo/ucsc_litreview/cytof/src/model3/sims/data/cytof_cb.rds";
R"cbData_R = readRDS(file=$cbDataPath)";
markers = R"colnames(cbData_R[[1]])";
cbData = [Matrix{Union{Float64, Missing}}(reshape(yi .+ 0, size(yi))) for yi in R"cbData_R"];
R"rm(cbData_R)"

# Get Dimensions
I = length(cbData)
J = size(cbData[1], 2)
N = [size(y, 1) for y in cbData]

for i in 1:I
    idx_missing = isnan.(cbData[i])
    cbData[i][idx_missing] .= missing
end

