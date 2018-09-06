println("Loading Packges for Cytof5 test...")
using Cytof5
using Test
using LinearAlgebra
using Distributions
using RCall
import Random
println("Starting Tests for Cytof5 test...")

include("MCMC_tests.jl")
