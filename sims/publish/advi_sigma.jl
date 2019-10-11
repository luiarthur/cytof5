using Distributions
using Cytof5
using Flux
using BSON

# Plotting
using PyCall
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
matplotlib.use("Agg")
# Load current dir
pushfirst!(PyVector(pyimport("sys")."path"), "../vb")
plot_yz = pyimport("plot_yz")
blue2red = pyimport("blue2red")


# Directory containing CB results 
data_dir = "../../sims/vb/results/vb-cb-paper/test/"

# Path to mm1 and best output
path_to_vb_cb_output = "$(data_dir)/output.bson"

# Get output for miss-mech-0 and miss-mech-1
output = BSON.load(path_to_vb_cb_output)

# Posterior Samples
samps = [Cytof5.VB.rsample(output[:state])[2] for _ in 1:100]
sig2s = [s.sig2 for s in samps]
Zs = [reshape(s.v, 1, length(s.v)) .> s.H for s in samps]
Ws = cat([s.W for s in samps]..., dims=3)

