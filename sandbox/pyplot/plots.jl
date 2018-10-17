import PyPlot
const plt = PyPlot
printKeys(x) = [print("$k  ") for k in keys(x)]
using Distributions
#=
plt.ioff() # turn off interactive graphics. turn on with plt.ion()
=#

function pyRange(n::Int)
  return collect(0:(n-1))
end


function plotZ(Z::Matrix{T}) where {T <: Number}
  J, K = size(Z)
  cm = plt.cm_get_cmap(:Greys)
  img = plt.imshow(Z, aspect="auto", vmin=0, vmax=1, cmap=cm);
  ax = plt.gca()
  [ plt.axhline(y=i+.5, color="grey", linewidth=.5) for i in pyRange(J)];
  [ plt.axvline(x=i+.5, color="grey", linewidth=.5) for i in pyRange(K)];
  plt.yticks(pyRange(J), 1:J);
  plt.xticks(pyRange(K), 1:K, rotation=:vertical);
  return img
end

function plotY(Y::Matrix{T}, clim=(-3,3)) where {T  <: Number}
  cm = plt.cm_get_cmap(:bwr)
  cm[:set_under](color=:blue)
  cm[:set_over](color=:red)
  cm[:set_bad](color=:black)
  img = plt.imshow(Y, vmin=clim[1], vmax=clim[2], aspect="auto", cmap=cm)
  plt.plt[:colorbar]();
  return img
end

### IBP ###
J = 32
K = 8
Z = Int.(rand(J, K) .> .6);

plotZ(Matrix(Z'))
plt.savefig("img/Z.pdf")
plt.close()

### Heatmap ###
Y = randn(30000, 32) * 2;
Y[:, 3] .= NaN;

plotY(Y)
plt.savefig("img/heatmap.pdf")
plt.close()

# yZ plot
plt.subplot2grid((1, 10), (0, 0), colspan=3)
plotZ(Matrix(Z))
plt.xticks(rotation=:horizontal)
plt.subplot2grid((1, 10), (0, 3), colspan=7)
plotY(Matrix(Y'));
plt.yticks(pyRange(J), 1:J);
plt.xticks(rotation=:vertical)
plt.tight_layout()
plt.savefig("img/yZ.pdf")

# kde
function kde(x::Vector{T}; from::T=minimum(x), to::T=maximum(x), numPoints::Int=10000,
             bw::Float64=0.0, kernel::Function=z->pdf(Normal(), z)) where {T <: Number}
  n = length(x)
  if bw == 0.0
    bw = std(x) * n ^ (-0.2) * 1.06
  end
  x_grid = collect(range(from, stop=to, length=numPoints))
  return Dict(:dx => [sum(kernel.((xi .- x) / bw)) / (n * bw) for xi in x_grid], :x => x_grid)
end

dist = Gamma(3, 200)
@time x = rand(dist, 10000);
@time d = kde(x, numPoints=100, bw=100.);
plt.plot(d[:x], pdf.(dist, d[:x]), label=:truth, color=:blue)
plt.plot(d[:x], d[:dx], color=:orange, label=:kde)
x_ci = quantile(x, [.025, .975])
plt.fill_between(d[:x][x_ci[1] .< d[:x] .< x_ci[2]], d[:dx][x_ci[1] .< d[:x] .< x_ci[2]],
                color=:orange)
plt.tight_layout()
plt.legend()
plt.savefig("img/den.pdf")


# Axis everywhere
# https://matplotlib.org/examples/api/two_scales.html
fig, ax1 = plt.subplots()
ax1[:plot](randn(100))
ax1[:set_xlabel]("x")
# Make the y-axis label, ticks and tick labels match the line color.
ax1[:set_ylabel](:leftLabel, color=:blue)
ax1[:tick_params]("y", colors=:blue)

ax2 = ax1[:twinx]()
ax2[:set_ylabel](:blabla)
ax2[:tick_params](:blable)
plt.savefig("img/axisGalore.pdf")
