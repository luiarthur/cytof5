using BSON
using RCall
@rimport graphics as plt
@rimport grDevices as dev
@rimport cytof3

# For BSON
using Cytof5, Flux, Distributions

if length(ARGS) == 0
  OUTPUT_PATH = "results/vb-sim-paper/01/out.bson"
else
  OUTPUT_PATH = parse(Int, ARGS[1])
end

out = BSON.load(OUTPUT_PATH)
RESULTS_DIR = join(split(OUTPUT_PATH, "/")[1:end-1], "/")
IMG_DIR = "$(RESULTS_DIR)/img/"
mkpath(IMG_DIR)

c = out[:c]
elbo = out[:metrics][:elbo] / sum(c.N)
state = out[:state]

plt.plot(elbo, xlab="", ylab="", typ="l")

NSAMPS = 100
samples = [Cytof5.VB.rsample(state)[2] for n in 1:NSAMPS]



Z = [Int.(reshape(s.v, 1, c.K) .> s.H) for s in samples]
mean_Z = mean(Z).data
dev.pdf("$(IMG_DIR)/Z.pdf")
cytof3.my_image(mean_Z, xlab="features", ylab="markers")
plt.abline(h=collect(1:c.J) .+ .5, v=collect(1:c.K) .+ .5, col="grey")
dev.dev_off()


W = [s.W.data for s in samples]
dev.pdf("$(IMG_DIR)/W.pdf")
for i in 1:c.I
  Wi = hcat([w[i, :] for w in W]...)
  plt.boxplot(Wi');
end
dev.dev_off()
