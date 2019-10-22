import Base.show

function padZeroCols(x::Matrix, desiredSize::Int)
  @assert size(x, 2) <= desiredSize

  n = size(x, 1)

  if size(x, 2) == desiredSize
    return x
  else
    return padZeroCols([x zeros(n)], desiredSize)
  end
end

#= Test
padZeroCols(randn(3,5), 10)
=#

"""
Do `expr` if `condition` holds.
"""
macro doIf(condition, expr)
  return quote
    if $(esc(condition))
      $(esc(expr))
    end
  end
end

# Pretty printing of InverseGamma
function show(io::IO, x::Distributions.InverseGamma)
  print(io, "InverseGamma(shape=$(shape(x)), scale=$(scale(x)))")
end

# Pretty printing of Gamma
function show(io::IO, x::Distributions.Gamma)
  print(io, "Gamma(shape=$(shape(x)), rate=$(rate(x)))")
end


"""
log info with println and flush
"""
function logger(x; newline=true)
  if newline
    println(x)
  else
    print(x)
  end
  flush(stdout)
end

"""
solve for inverse gamma parameters
"""
function solve_ig_params(; mu::Float64, sig2::Float64)
  @assert mu > 0
  @assert sig2 > 0
  a = (mu^2 / sig2) + 2
  b = mu * (a - 1)
  return (a, b)
end

"""
sample from inverse gamma with an upper truncation point
"""
function rand_uptrunc_ig(ig::InverseGamma, upper::Float64)
  lg_u_max = logcdf(ig, upper) 
  lg_u = log(rand()) + lg_u_max
  init = upper / 2.0
  return invlogcdf(ig, lg_u)
end

#= Test
ig = InverseGamma(1002, 1001)
mean(ig), std(ig)
@time x = [rand_uptrunc_ig(ig, .3) for i in 1:10000];
=#


"""
Read CB data from a text file.
Text file must have following properties:
- first line is: "N: N1 N2 N3"
- second line is the name of the markers separated by a single space
- the remainder of the file is the marker expression levels for each cell 
  by sample. No line breaks are necessary between samples. Each line should
  contain expression levels for each cell equal to the number of markers.


Usage
=====

y  = loadCB(path_to_data)
"""
function loadCB(path_to_data; ElType::Type=Float64)
  return open(path_to_data) do f
    # read number of cells in first line
    N = readline(f)
    N = split(N)[2:end]
    N = parse.(Int, N)

    # Number of samples
    I = length(N)

    # Read marker names
    markers = split(readline(f))

    # Read data into separate samples
    y = [begin
           yi = Vector{ElType}[]
           for _ in 1:N[i]
             line = split(readline(f))
             line = [replace(obs, "NA" => "NaN") for obs in line]
             line = parse.(ElType, line)
             append!(yi, [line])
           end
           Matrix(hcat(yi...)')
         end for i in 1:I]

    return y
  end
end

