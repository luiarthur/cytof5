using Revise
using Cytof5
using Random
using Distributions
using PyPlot
using BSON
include("../Util/Util.jl")
include("simulatedata.jl")

Random.seed!(10)
printDebug = false
println("Threads: $(Threads.nthreads())")
println("pid: $(getpid())")

simdat = simulatedata1(Z=zeros(7, 3), seed=0)
