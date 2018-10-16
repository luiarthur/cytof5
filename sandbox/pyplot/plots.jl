import PyPlot
const plt = PyPlot
printKeys(x) = [print("$k  ") for k in keys(x)]
#=
plt.ioff() # turn off interactive graphics. turn on with plt.ion()
=#

### IBP ###
J = 32
K = 8
Z = Int.(rand(J, K) .> .6);

plt.imshow(Z, aspect="auto", vmin=.01, vmax=.99);
ax = plt.gca()
cm = plt.cm_get_cmap()
cm[:set_under](color="white")
cm[:set_over](color="black")
[ plt.axhline(y=i+.5, color="grey", linewidth=.5) for i in 0:(J-1)];
[ plt.axvline(x=i+.5, color="grey", linewidth=.5) for i in 0:(K-1)];
plt.xticks(0:(K-1), 1:K);
plt.yticks(0:(J-1), 1:J);
plt.savefig("img/Z.pdf")
plt.close()

### Heatmap ###
x = randn(30000, 32) * 2;
x[:, 3] .= NaN;

cm = plt.cm_get_cmap(:bwr)
cm[:set_under](color=:blue)
cm[:set_over](color=:red)
cm[:set_bad](color=:black)
plt.imshow(x, vmin=-2, vmax=2, aspect="auto", cmap=cm)
plt.plt[:colorbar]();

plt.savefig("img/heatmap.pdf")

# yZ plot
plt.figure()
