import PyPlot
const plt = PyPlot
printKeys(x) = [print("$k  ") for k in keys(x)]
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
