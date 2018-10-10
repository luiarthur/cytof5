import PyPlot
const pplt = PyPlot
printKeys(x) = [print("$k  ") for k in keys(x)]
#pplt.ioff() # turn off interactive graphics. turn on with pplt.ion()

### IBP ###
J = 32
K = 8
Z = Int.(rand(J, K) .> .6);

pplt.imshow(Z, aspect="auto", vmin=.01, vmax=.99);
ax = pplt.gca()
cm = pplt.cm_get_cmap()
cm[:set_under](color="white")
cm[:set_over](color="black")
[ pplt.axhline(y=i+.5, color="grey", linewidth=.5) for i in 0:(J-1)];
[ pplt.axvline(x=i+.5, color="grey", linewidth=.5) for i in 0:(K-1)];
pplt.xticks(0:(K-1), 1:K);
pplt.yticks(0:(J-1), 1:J);
pplt.savefig("img/Z.pdf")

### Heatmap ###
x = randn(300, 32) * 2;
x[:, 3] .= NaN;

pplt.imshow(x, vmin=-2, vmax=2, aspect="auto", cmap="bwr")
pplt.plt[:colorbar]();
cm = pplt.cm_get_cmap()
cm[:set_under](color="red")
cm[:set_over](color="blue")
cm[:set_bad](color="black") # WTF?
pplt.savefig("img/heatmap.pdf")
