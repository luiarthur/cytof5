using BSON
using RCall
@rimport graphics as plt
@rimport grDevices as dev
@rimport cytof3
@rimport rcommon

# For BSON
using Cytof5, Flux, Distributions

if length(ARGS) == 0
  # OUTPUT_PATH = "results/vb-sim-paper/01/output.bson"
  OUTPUT_PATH = "results/vb-sim-paper/test/0/output.bson"
else
  OUTPUT_PATH = parse(Int, ARGS[1])
end

out = BSON.load(OUTPUT_PATH)
simdat = out[:simdat]
RESULTS_DIR = join(split(OUTPUT_PATH, "/")[1:end-1], "/")
IMG_DIR = "$(RESULTS_DIR)/img/"
mkpath(IMG_DIR)

c = out[:c]
elbo = out[:metrics][:elbo] / sum(c.N)
state = out[:state]
state_hist = out[:state_hist]
metrics = out[:metrics]

# ELBO
dev.pdf("$(IMG_DIR)/elbo.pdf")
plt.plot(elbo[200:end], xlab="", ylab="", typ="l")
dev.dev_off()

# Metrics
dev.pdf("$(IMG_DIR)/metrics.pdf")
plt.par(mfrow=[length(metrics), 1], oma=rcommon.oma_ts(), mar=rcommon.mar_ts())
for (k, m) in metrics
  plt.plot(metrics[k][1:end]/sum(c.N), xlab="iter", ylab=string(k), typ="l",
           xaxt="n")
end
plt.axis(1)
plt.par(mfrow=[1, 1], oma=rcommon.oma_default(), mar=rcommon.mar_ts())
dev.dev_off()

NSAMPS = 200
samples = [Cytof5.VB.rsample(state)[2] for n in 1:NSAMPS]
trace = [Cytof5.VB.rsample(s)[2] for s in state_hist]

# y_samps
m = [isnan.(yi) for yi in simdat[:y]]
y_samps = [Tracker.data.(Cytof5.VB.rsample(state, simdat[:y], c)[3]) for n in 1:10]
dev.pdf("$(IMG_DIR)/y_hist.pdf")
for i in 1:c.I
  plt.hist(vec(y_samps[end][i][m[i]]), xlab="", ylab="", main="");
end
dev.dev_off()

# Z
if c.use_stickbreak
  Z = [Int.(cumprod(reshape(s.v, 1, c.K)) .> s.H) for s in samples]
else
  Z = [Int.(reshape(s.v, 1, c.K) .> s.H) for s in samples]
end
mean_Z = mean(Z).data
dev.pdf("$(IMG_DIR)/Z.pdf")
cytof3.my_image(mean_Z, xlab="features", ylab="markers", col=cytof3.greys(10), addL=true)
plt.abline(h=collect(1:c.J) .+ .5, v=collect(1:c.K) .+ .5, col="grey")
dev.dev_off()

# W
W = [s.W.data for s in samples]
dev.pdf("$(IMG_DIR)/W.pdf")
for i in 1:c.I
  Wi = hcat([w[i, :] for w in W]...)
  plt.boxplot(Wi');
end
dev.dev_off()

# mu
mu = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in samples]...)
dev.pdf("$(IMG_DIR)/mu.pdf")
plt.boxplot(mu');
plt.abline(h=0, v=c.L[0]+.5, col="grey");
dev.dev_off()

# sig2
sig2 = hcat([s.sig2.data for s in samples]...)
dev.pdf("$(IMG_DIR)/sig2.pdf")
plt.boxplot(sig2');
dev.dev_off()

# alpha
alpha = vcat([s.alpha.data for s in samples]...);
dev.pdf("$(IMG_DIR)/alpha.pdf")
rcommon.plotPost(alpha, main="alpha");
dev.dev_off()


### trace plots ###
mkpath("$(IMG_DIR)/trace/")

# Z trace
if c.use_stickbreak
  Z_trace = [Int.(reshape(cumprod(t.v), 1, c.K) .> t.H).data for t in trace]
else
  Z_trace = [Int.(reshape(t.v, 1, c.K) .> t.H).data for t in trace]
end
dev.pdf("$(IMG_DIR)/trace/Z.pdf")
for z in Z_trace
  cytof3.my_image(z, xlab="features", ylab="markers")
  plt.abline(h=collect(1:c.J) .+ .5, v=collect(1:c.K) .+ .5, col="grey")
end
dev.dev_off()

# mu_trace
mu_trace = hcat([[-cumsum(s.delta0.data); cumsum(s.delta1.data)] for s in trace]...)
dev.pdf("$(IMG_DIR)/trace/mu.pdf")
plt.matplot(mu_trace', xlab="iter", ylab="mu", typ="l", lwd=2)
dev.dev_off()

# sig2_trace
sig2_trace = hcat([s.sig2.data for s in trace]...)
dev.pdf("$(IMG_DIR)/trace/sig2.pdf")
plt.matplot(sig2_trace', xlab="iter", ylab="sig2", typ="l", lwd=2)
dev.dev_off()

# W trace
W_trace = cat([s.W.data for s in trace]..., dims=3)
for i in 1:c.I
  dev.pdf("$(IMG_DIR)/trace/W$(i).pdf")
  plt.matplot(W_trace[i, :, :]', xlab="iter", ylab="W$(i)", typ="l", lwd=2)
  dev.dev_off()
end

# alpha trace
alpha_trace = vcat([s.alpha.data for s in trace]...)
dev.pdf("$(IMG_DIR)/trace/alpha.pdf")
plt.plot(alpha_trace, xlab="iter", ylab="alpha", typ="l", lwd=2)
dev.dev_off()

