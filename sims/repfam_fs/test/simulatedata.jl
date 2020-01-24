"""
Z1 is a binary matrix with very different phenotypes
"""
Z1 = let
  Z = zeros(Bool, 7, 3)
  Z[1:3, 1] .= 1
  Z[4:6, 2] .= 1
  Z[[1, 4, 7], 3] .= 1
  Z
end


"""
Z2 is a binary matrix with two similar phenotypes
"""
Z2 = let
  Z = ones(Bool, 7, 3)
  Z[[4, 7], 1] .= 0
  Z[[1, 4, 7], 2] .= 0
  Z[2:6, 3] .= 0
  Z
end


"""
Z3 is a binary matrix with several similar phenotypes
"""
Z3 = let
  Z = [Z2 ones(Bool, 7, 2)]
  Z[end, 4] = false
  Z[[2, 4, 7], 5] .= false
  Z
end


"""
Z4 is a binary matrix where the first two columns are similar
(off by one bit), and the other three columns are very different
from every other column in the matrix. The three columns 
are more similar to one another.
"""
Z4 = let
  J, K = size(Z3)
  num_new_rows = 20 - J
  new_block = zeros(Bool, num_new_rows, K)
  new_block[1:div(num_new_rows, 2), 1:2] .= 1
  new_block[(div(num_new_rows, 2) + 1):end, 3:end] .= 1
  [Z3; new_block]
end

# All Zs.
Zs = [Z1, Z2, Z3, Z4]



function simulatedata1(; Z, N=[300, 300], L=Dict(0=>1, 1=>1),
                       mus=Dict(0=>[-2.0], 1=>[2.0]),
                       W=Matrix([[.7, 0, .3] [.6, .1, .3]]'),
                       sig2=[.1, .1], propmissingscale=0.3,
                       beta=[-9.2, -2.3],  # linear missing mechanism
                       sortLambda=false, seed=nothing)
  if seed != nothing
    Random.seed!(seed)
  end

  J, K = size(Z)
  I = length(N)

  @assert size(W) == (I, K)
  @assert length(sig2) == I
  @assert all(sig2 .> 0)
  @assert all(length(mus[key]) == L[key] for key in keys(L))
  @assert all(N .> 0)

  # Simulate cell phenotypes (lambda)
  lam = [begin
           rand(Categorical(W[i, :]), N[i])
         end for i in 1:I]

  if sortLambda
    lam = [sort(lami) for lami in lam]
  end

  # Generate y
  y = [zeros(Ni, J) for Ni in N]
  y_complete = deepcopy(y)

  z(i::Int, n::Int, j::Int) = Z[j, lam[i][n]]

  mu(i::Int, n::Int, j::Int) = mus[z(i, n, j)][1]  # NOTE: only for L[0]=L[1]=1!
  sig(i::Int) = sqrt(sig2[i])

  for i in 1:I
    sig_i = sig(i)
    for j in 1:J
      for n in 1:N[i]
        y_complete[i][n, j] = rand(Normal(mu(i, n, j), sig_i))
      end
      # Set some y to be missing
      # Using a linear missing mechanism to rate missingness
      p_miss = [Cytof5.Model.prob_miss(y_complete[i][n, j], beta)
                for n in 1:N[i]]
      # If this is not used, then many positive markers may be missing too.
      prop_not_expressed = sum(W[i,:] .* (1 .- Z[j,:]))
      # If propmissingscale is 1, about all the non-expressed observations
      # will be missing. Best to set propmissingscale to about 0.3.
      prop_missing = propmissingscale * prop_not_expressed

      num_missing = Int(round(N[i] * prop_missing))
      idx_missing = Distributions.wsample(1:N[i], p_miss, num_missing,
                                          replace=false)
      y[i][:, j] .= y_complete[i][:, j] .+ 0
      y[i][idx_missing, j] .= NaN
    end
  end


  return Dict(:Z => Z, :N => N, :L => L, :mus => mus, :W => W, :seed => seed,
              :lam => lam, :sig2 => sig2, :y => y, :y_complete => y_complete,
              :beta => beta)
end
