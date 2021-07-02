import Pkg; Pkg.activate("../../")
using BSON

include("salso/salso.jl")

function bytes2float64(byte::Vector{UInt8})
    n = length(byte)
    @assert mod(n, 8) == 0
    counter = 1
    out = Float64[]
    while counter < n
        idx = counter:counter+7
        x = reinterpret(Float64, byte[idx])
        append!(out, x)
        counter += 8
    end
    return out
end

function get_Z_lam(path)
    results = BSON.parse(path)
    num_samples = length(results[:out][1])

    Zs = [let
         _z = o[:Z]
         J, K = _z[:size]
         reshape(Bool.(_z[:data]), J, K)
     end for o in results[:out][1]]

    Ws = [let
         _w = o[:W]
         I, K = _w[:size]
         reshape(bytes2float64(_w[:data]), I, K)
     end for o in results[:out][1]]


    lams = [let
         _lam = o[:lam]
         [Int.(lami[:data]) for lami in _lam]
     end for o in results[:out][1]]

    I = length(lams[1])
    _argbest = [argbest(Zs, [W[i, :] for W in Ws]) for i in 1:I]

    return (Z=[Zs[_argbest[i]] for i in 1:I],
            lam=[lams[_argbest[i]][i] for i in 1:I],
            W=[Ws[_argbest[i]][i, :] for i in 1:I])
end


function write_summary(outdir, summary)
    mkpath(outdir)
    I = length(summary.Z)

    for i in 1:I
        open("$(outdir)/Z$(i).txt", "w") do io
            writedlm(io, Int.(summary.Z[i]))
        end
        open("$(outdir)/lam$(i).txt", "w") do io
            writedlm(io, summary.lam[i])
        end
        open("$(outdir)/w$(i).txt", "w") do io
            writedlm(io, summary.W[i])
        end
    end
end
