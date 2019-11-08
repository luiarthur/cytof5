#=
using Revise, Test
=#
using Cytof5
using Random
using RCall, Distributions
using BSON

Random.seed!(10)
printDebug = false
println("Threads: $(Threads.nthreads())")
println("pid: $(getpid())")

@rimport graphics
@rimport grDevices

function init_state_const_data(; N=[300, 200, 100], J=8, K=4,
                               L=Dict(0=>5, 1=>3))

  I = length(N)
  simdat = Cytof5.Model.genData(J, N, K, L, sortLambda=true)
  d = Cytof5.Model.Data(simdat[:y])
  c = Cytof5.Model.defaultConstants(d, K * 2, Dict(0=>5, 1=>5))
  s = Cytof5.Model.genInitialState(c, d)
  t = Cytof5.Model.Tuners(d.y, c.K)
  X = Cytof5.Model.eye(Float64, d.I)

  return Dict(:d => d, :c => c, :s => s, :t => t, :X => X,
              :simdat => simdat)
end


@testset "update W*, r, omega" begin
  config = init_state_const_data() 
  cfs = Cytof5.Model.ConstantsFS(config[:c])
  dfs = Cytof5.Model.DataFS(config[:d], config[:X])
  sfs = Cytof5.Model.StateFS{Float64}(config[:s], dfs)
  tfs = Cytof5.Model.TunersFS(config[:t], config[:s], config[:X])

  # Do one update for W_star, r, omega.
  println("r init: $(sfs.r)")
  println("W_star init: $(sfs.W_star)")
  println("omega init: $(sfs.omega)")

  Cytof5.Model.update_W_star!(sfs, cfs, dfs, tfs)
  Cytof5.Model.update_r!(sfs, cfs, dfs)
  Cytof5.Model.update_omega!(sfs, cfs, dfs, tfs)

  println("W_star after: $(sfs.W_star)")
  println("W after: $(sfs.theta.W)")
  println("r after: $(sfs.r)")
  println("omega after: $(sfs.omega)")

  Cytof5.Model.printConstants(cfs)

  # Fit model.
  # For algo tests
  # nmcmc = 100
  # nburn = 1200
  # For compile tests
  nmcmc = 3
  nburn = 5
  out = Cytof5.Model.fit_fs!(sfs, cfs, dfs, tuners=tfs,
                             nmcmc=nmcmc, nburn=nburn,
                             printFreq=1, time_updates=true,
                             computeDIC=true, computeLPML=true, Z_thin=5)
  BSON.bson("result/out_fs.bson", out)
  BSON.bson("result/data_fs.bson", Dict(:simdat => config[:simdat]))
end

#= READ
@rimport cytof3
@rimport rcommon
extract(chain, sym) = [samp[sym] for samp in chain]
out = BSON.load("result/out_fs.bson")
println("Number of samples: $(length(out[:samples][1]))")
simdat = BSON.load("result/data_fs.bson")[:simdat]
graphics.plot(out[:loglike], ylab="loglike", xlab="iter", main="", typ="l")
cytof3.my_image(out[:lastState].theta.Z)
cytof3.my_image(simdat[:Z])
Wstars = cat(extract(out[:samples][1], :W_star)..., dims=3)
rs = cat(extract(out[:samples][1], :r)..., dims=3)
Ws = Wstars .* rs ./ sum(Wstars .* rs, dims=2)
omegas = Matrix(hcat(extract(out[:samples][1], :omega)...)')
rcommon.plotPosts(omegas)
sig2s = Matrix(hcat(extract(out[:samples][1], :theta__sig2)...)')
graphics.boxplot(Ws[1, :, :]')
graphics.abline(h=simdat[:W][1, :])
graphics.boxplot(Wstars[1, :, :]')
mean(Wstars, dims=3)
mean(rs, dims=3)
std(rs, dims=3)
rcommon.plotPosts(sig2s)
simdat[:sig2]
Cytof5.Model.compute_p([0., 0., 1.], omegas[end, :])
Zs = cat(extract(out[:samples][1], :theta__Z)..., dims=3)
cytof3.my_image(dropdims(mean(Zs, dims=3), dims=3))
graphics.plot(mean(rs, dims=3)[1, :, :], var"type"="o")
=#


@testset "Compile Model.State." begin
  I = 3
  J = 8
  K = 5
  N = [3, 1, 2] .* 10
  L = Dict{Int, Int}(0 => 5, 1 => 3)

  import Cytof5.Model
  import Cytof5.Model.Cube

  state = Model.State{Float16}()
  state.Z=Matrix{Bool}(undef, J, K)
  state.delta=Dict{Bool, Vector{Float16}}()
  state.alpha=Float16(1.0)
  state.v=ones(Float16, K)
  state.W=rand(Float16, I, K)
  state.sig2=Vector{Float16}(undef, I)
  state.eta=Dict{Bool, Cube{Float16}}()
  state.lam=[ones(Int8, N[i]) for i in 1:I]
  state.gam=[ones(Int8, N[i], J) for i in 1:I]
  state.y_imputed=[randn(Float16, N[i], J) for i in 1:I]
  state.eps=[Float16(.05) for i in 1:I]
  @test true

  # Debug Data constructor
  y = [ randn(N[i], J) for i in 1:I ]
  data = Model.Data(y)
  @test data.I == I
  @test data.J == J
  @test data.N == N

  # Debug Constants constructor
  constants = Model.defaultConstants(data, K, L)
  @test constants.K == K
  @test constants.L == L
end


@testset "Compile Model.genData." begin
  J = 8
  N = [3, 1, 2] * 100
  I = length(N)
  K = 4
  L = Dict{Int, Int}(0 => 5, 1 => 3)
  @time dat = Cytof5.Model.genData(J, N, K, L)

  plot_dat = R"cytof3::plot_dat"
  myImage = R"cytof3::my.image"

  grDevices.pdf("result/simdat_test.pdf")
  myImage(dat[:Z])
  myImage(Cytof5.Model.genZ(J, K, .6))

  for i in 1:I
    for j in 1:J
      plot_dat(dat[:y_complete], i, j)
    end
  end
  grDevices.dev_off()
  @test true
end


@testset "Compile Model.genInitialState." begin
  J = 8
  N = [3, 1, 2] * 100 # Super fast even for 10000 obs. 
  I = length(N)
  K = 4
  # L = Dict{Int, Int}(0 => 5, 1 => 3)
  L = Dict{Int, Int}(0 => 4, 1 => 4)
  @time dat = Cytof5.Model.genData(J, N, K, L, sortLambda=true)
  y_dat = Cytof5.Model.Data(dat[:y])

  R"""
  library(cytof3)
  """

  K_MCMC = K #10
  L_MCMC = L #5
  @time c = Cytof5.Model.defaultConstants(y_dat, K_MCMC, L_MCMC,
                                          noisyDist=Normal(0, sqrt(10)),
                                          yBounds=[-5., -3.5, -2.])
  Cytof5.Model.printConstants(c)
  # Plot miss mech
  R"pdf('result/beta.pdf')"
  y_grid = collect(range(-10, stop=5, length=300))
  R"par(mfrow=c($(y_dat.I), 1))"
  for i in 1:y_dat.I
    p = [Cytof5.Model.prob_miss(yy, c.beta[:, i]) for yy in y_grid]
    R"plot"(y_grid, p, main="i: $i", typ="l",
            xlab="y", ylab="prob of missing", xlim=[-10, 5])
  end
  R"par(mfrow=c(1,1))"
  R"dev.off()"


  # @time init = Cytof5.Model.genInitialState(c, y_dat)
  @time init = Cytof5.Model.smartInit(c, y_dat)
  println("init delta: $(init.delta)")

  printstyled("Test Model Fitting...\n", color=:yellow)
  @time out, lastState, ll, metrics, dden=Cytof5.Model.cytof5_fit(
    init, c, y_dat,
    nmcmc=200, nburn=200,
    computeLPML=true,
    computeDIC=true,
    computedden=true,
    joint_update_Z=false)
  println("Type of dden: $(typeof(dden[end]))")

  println("Type of output: $(typeof(out[1])))")
  Zpost = [o[:Z] for o in out[1]]
  R"pdf('result/Z_post_mean.pdf')"
  R"image"(1 .- mean(Zpost))
  R"dev.off()"

  # @save "result/out.jld2" out dat ll lastState  # requires JLD2, FileIO

  BSON.bson("result/out.bson", Dict(
    :out => out,
    :dat => dat,
    :ll => ll,
    :lastState => lastState))

  @test true
end
