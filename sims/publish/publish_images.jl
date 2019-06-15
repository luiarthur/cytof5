using Distributions
using Cytof5, Random
using JLD2, FileIO
import BSON

# Plotting
using PyCall
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
matplotlib.use("Agg")
# Load current dir
pushfirst!(PyVector(pyimport("sys")."path"), "../vb")
plot_yz = pyimport("plot_yz")
blue2red = pyimport("blue2red")
# multiple pages
PdfPages = pyimport("matplotlib.backends.backend_pdf").PdfPages

PATH_TO_RESULTS = "/scratchdata/cytof/results/"

# plot y/z
#
