import Pkg; Pkg.activate("../../")
using BSON

include("salso/salso.jl")

function frombytes(byte::Vector{UInt8}, T)
    n = length(byte)
    @assert mod(n, 8) == 0
    counter = 1
    out = T[]
    while counter < n
        idx = counter:counter+7
        x = reinterpret(T, byte[idx])
        append!(out, x)
        counter += 8
    end
    return out
end

bytes2int8(byte::Vector{UInt8}) = frombytes(byte, Int8)
bytes2int64(byte::Vector{UInt8}) = frombytes(byte, Int64)
bytes2float64(byte::Vector{UInt8}) = frombytes(byte, Float64)

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

function get_data(path; I=3)
    data = BSON.parse(path)
    y = [let
             yvec = bytes2float64(data[:simdat][:y][i][:data])
             size = data[:simdat][:y][i][:size]
             reshape(yvec, size...)
         end for i in 1:I]

    y_complete = [let
             yvec = bytes2float64(data[:simdat][:y_complete][i][:data])
             size = data[:simdat][:y_complete][i][:size]
             reshape(yvec, size...)
         end for i in 1:I]

    lam = [bytes2int64(data[:simdat][:lam][i][:data])
           for i in 1:I]

    return (y=y, y_complete=y_complete, lam=lam)
end
