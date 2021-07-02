import LinearAlgebra: diagm
import Statistics: mean
BinaryMat = Union{Matrix{Bool}, BitMatrix}

"""
Pairwise allocation matrix

pam(Z::BinaryMat) = Z * Z'
"""
pam(Z::BinaryMat) = Z * Z'
pam(Z::BinaryMat, w::AbstractVector{<:Real}) = Z * diagm(w) * Z'
pam(Zs::Vector{<:BinaryMat}, ws::Vector{<:AbstractVector{<:Real}}) = [pam(Z, w) for (Z, w) in zip(Zs, ws)]

function argbest(Zs::Vector{<:BinaryMat}, ws::Vector{<:AbstractVector{<:Real}})
    pams = pam(Zs, ws)
    epam = mean(pams)
    diffs = [sum((pam - epam) .^ 2) for pam in pams] 
    return argmin(diffs)
end
