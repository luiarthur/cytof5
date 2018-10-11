println("Loading Packages")
using Plots
pyplot()

#= References:
http://docs.juliaplots.org/latest/colors/#colorbrewer
=#

### IBP ###
println("Gen Z")
J = 32
K = 8
Z = Int.(rand(J, K) .> .8);

println("Plot Z")
heatmap(Z, c=:Greys, legend=:none)
plot!((1:K) .- .5, linetype=:vline, c=:lightgrey)
plot!((1:J) .- .5, linetype=:hline, c=:lightgrey)
savefig("img/Z.pdf")

### Heatmap ###
println("gen x")
x = randn(300, 32) * 2;
x[:, 3] .= NaN;

println("plot x")
heatmap(x, c=:pu_or, clim=(-3,3), background_color_inside=:black)
savefig("img/heatmap.pdf")
