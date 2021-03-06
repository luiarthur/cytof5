using JLD2, FileIO
using RCall
using Distributions

function loadSingleObj(objPath)
  data = load(objPath)
  return data[collect(keys(data))[1]]
end

include("PreProcess.jl")
include("../sim_study/util.jl")
cbDataPath = "data/cytof_cb_with_nan.jld2"
cbReducedDataPath = "data/reduced_cb.jld2"
table = R"table"

cbData = loadSingleObj(cbDataPath)
# Reduce Data by removing highly non-expressive / expressive columns
goodColumns, J = PreProcess.preprocess!(cbData, maxNanOrNegProp=.9, maxPosProp=.9)

@save cbReducedDataPath deepcopy(cbData)

# Histograms
legend = R"legend"
R"require(rcommon)"

I = length(cbData)
J_new = size(cbData[1], 2)

mar_tmp = [2, 5.1, 2, 2.1]
imgDir = "results/misc/img/"
mkpath(imgDir)

println("Create index of missing obs")
for i in 1:I
  idx_missing = findall(isnan.(cbData[i]))
  open("$(imgDir)/index_of_missing_obs_sample$(i).txt", "w") do f
    for idx in idx_missing
      n, j = (idx[1], idx[2])
      write(f, "i: $i, n: $n, j: $j\n")
    end
  end
end

util.plotPdf("$(imgDir)/reducedDataHist.pdf");
R"par(mfrow=c(4, 2), mar=$mar_tmp, cex.main=.8)";
for i in 1:I
  for j in 1:J_new
    yij = cbData[i][:, j]
    yij_obs = filter(x -> !isnan(x), yij)
    n = length(yij)
    util.hist(yij, ylab="density", main="sample $i, marker $(findall(goodColumns)[j])",
              xlab="", border="transparent", col="steelblue",
              nclass=80, prob=true, xlim=[-10, 4])
    prop_missing = mean(isnan.(yij))
    ymean = mean(yij_obs)
    ymean_pos = mean(filter(x -> x > 0, yij_obs))
    ymean_neg = mean(filter(x -> x < 0, yij_obs))
    ysd = std(yij_obs)
    ysd_pos = std(filter(x -> x > 0, yij_obs))
    ysd_neg = std(filter(x -> x < 0, yij_obs))
    legend("topleft", bty="n", cex=1, 
           legend=["prop. missing: $(round(prop_missing, digits=5))",
                   "num. obs: $(n)",
                   "mean: $(round(mean(yij_obs), digits=2))",
                   "mean (+): $(round(ymean_pos, digits=2))",
                   "mean (-): $(round(ymean_neg, digits=2))",
                   "sd: $(round(ysd, digits=2))",
                   "sd (+): $(round(ysd_pos, digits=2))",
                   "sd (-): $(round(ysd_neg, digits=2))"])
    util.abline(v=[ymean, ymean_pos, ymean_neg], col="red", lty=[1, 2, 2])
  end
end
R"par(mfrow=c(1, 1), oma=rcommon::oma.default(), mar=rcommon::mar.default(), cex.main=1)"
util.devOff()

# Count number of missing values per cell per sample
N = size.(cbData, 1)
num_missing = [[sum(isnan.(cbData[i][n, :])) for n in 1:N[i]] for i in 1:I]
for i in 1:I
  open("$(imgDir)/missing_values_per_cell_sample$(i).txt", "w") do f
    for n in 1:N[i]
      write(f, "$(num_missing[i][n])\n")
    end
  end
end

for i in 1:I
  util.plotPdf("$(imgDir)/missing_values_per_cell_sample$(i).pdf")
  util.plot(R"table($(num_missing[i])) / $(N[i])", 
            xlim=[0, J_new],
            ylab="Proportion of cells missing number of markers",
            xlab="number of missing marker expression levels")
  util.devOff()
end


