using BSON
using Cytof5

# For BSON
using Flux, Distributions

# Load sim data
SIMDAT_PATH = "../sim_study/simdata/kills-flowsom/N500/K5/90/simdat.bson"
simdat = BSON.load(SIMDAT_PATH)[:simdat]
simdat[:y] = Vector{Matrix{Float64}}(simdat[:y])

# Generate model constnats
c = Cytof5.VB.Constants(y=simdat[:y], K=10, L=Dict(false=>5, true=>3),
                        yQuantiles=[.0, .25, .5], pBounds=[.05, .8, .05],
                        use_stickbreak=false, tau=.005)

# Fit model
out = Cytof5.VB.fit(y=simdat[:y], niters=20000, batchsize=2000, c=c, nsave=30, seed=0)

# Save results
BSON.bson("results/out.bson", out)

#= Post process
using BSON, Cytof5, Flux, Distributions
using PyCall

plt = pyimport("matplotlib.pyplot")

out = BSON.load("results/out.bson")
c = out[:c]

length(out[:state_hist])

elbo = out[:metrics][:elbo] / sum(c.N)
plt.plot(elbo[500:end]); plt.show()

samples = [Cytof5.VB.rsample(out[:state])[2] for b in 1:100]
Z = [reshape(s.v, 1, c.K) .> s.H for s in samples]
Z_mean = mean(Z)

plt.imshow(Z_mean .> .5); plt.show()

=#
