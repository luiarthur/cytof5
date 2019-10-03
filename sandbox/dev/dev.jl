using Revise
using Cytof5

J = 32
N = [300, 100, 200]
K = 8
L = Dict(0 => 5, 1 => 3)

simdat = Cytof5.Model.genData(J, N, K, L)
d = Cytof5.Model.Data(simdat[:y])
c = Cytof5.Model.defaultConstants(d, K, L)
s = Cytof5.Model.genInitialState(c, d)
