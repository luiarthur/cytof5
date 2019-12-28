"""
printFreq: defaults to 0 => prints every 10%. turn off printing by 
           setting to -1.
"""
function cytof5_fit(init::State, c::Constants, d::Data;
                    nmcmc::Int=1000, nburn::Int=1000, 
                    monitors=[[:Z, :lam, :W, :v,
                               :sig2, :delta, :alpha, :v,
                               :eta, :eps]],
                    fix::Vector{Symbol}=Symbol[],
                    thins::Vector{Int}=[1],
                    thin_dden::Int=1,
                    printFreq::Int=0, flushOutput::Bool=false,
                    computeDIC::Bool=false, computeLPML::Bool=false,
                    computedden::Bool=false,
                    sb_ibp::Bool=false,
                    use_repulsive::Bool=false, joint_update_Z::Bool=false,
                    verbose::Int=1)

  if verbose >= 1
    fixed_vars_str = join(fix, ", ")
    if fixed_vars_str == ""
      fixed_vars_str = "nothing"
    end
    println("fixing: $fixed_vars_str")
    println("Use stick-breaking IBP: $(sb_ibp)")
  end

  @assert printFreq >= -1
  if printFreq == 0
    numPrints = 10
    printFreq = Int(ceil((nburn + nmcmc) / numPrints))
  end


  y_tuner = begin
    dict = Dict{Tuple{Int, Int, Int}, MCMC.TuningParam{Float64}}()
    for i in 1:d.I
      for n in 1:d.N[i]
        for j in 1:d.J
          if d.m[i][n, j] == 1
            dict[i, n, j] = MCMC.TuningParam(1.0)
          end
        end
      end
    end
    dict
  end

  tuners = Tuners(y_imputed=y_tuner, # yinj, for inj s.t. yinj is missing
                  Z=MCMC.TuningParam(MCMC.sigmoid(c.probFlip_Z, a=0.0, b=1.0)),
                  v=[MCMC.TuningParam(1.0) for k in 1:c.K])

  # Loglike
  loglike = Float64[]

  function printMsg(iter::Int, msg::String)
    if printFreq > 0 && iter % printFreq == 0
      print(msg)
    end
  end

  # Instantiate (but not initialize) CPO stream
  if computeLPML
    cpoStream = MCMC.CPOstream{Float64}()
  end

  # DIC
  if computeDIC
    local tmp = DICparam(p=deepcopy(d.y),
                         mu=deepcopy(d.y),
                         sig=[zeros(Float64, d.N[i]) for i in 1:d.I],
                         y=deepcopy(d.y))
    dicStream = MCMC.DICstream{DICparam}(tmp)

    function updateParams(d::MCMC.DICstream{DICparam}, param::DICparam)
      d.paramSum.p += param.p
      d.paramSum.mu += param.mu
      d.paramSum.sig += param.sig
      d.paramSum.y += param.y
      return
    end

    function paramMeanCompute(d::MCMC.DICstream{DICparam})::DICparam
      return DICparam(d.paramSum.p ./ d.counter,
                      d.paramSum.mu ./ d.counter,
                      d.paramSum.sig ./ d.counter,
                      d.paramSum.y ./ d.counter)
    end

    function loglikeDIC(param::DICparam)::Float64
      ll = 0.0

      for i in 1:d.I
        for j in 1:d.J
          for n in 1:d.N[i]
            y_inj_is_missing = (d.m[i][n, j] == 1)

            # NOTE: Refer to `../compute_loglike.jl` for reasoning.
            if y_inj_is_missing

              # Compute p(m_inj | y_inj, theta) term.
              ll += log(param.p[i][n, j])
            end

            # Compute p(y_inj | theta) term.
            ll += logpdf(Normal(param.mu[i][n, j], param.sig[i][n]),
                         param.y[i][n, j])
          end
        end
      end

      return ll
    end

    # FIXME: Doesn't work for c.noisyDist not Normal
    function convertStateToDicParam(s::State)::DICparam
      p = [[prob_miss(s.y_imputed[i][n, j], c.beta[:, i])
            for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]

      mu = [[s.lam[i][n] > 0 ? mus(i, n, j, s, c, d) : 0.0 
             for n in 1:d.N[i], j in 1:d.J] for i in 1:d.I]

      sig = [[s.lam[i][n] > 0 ? sqrt(s.sig2[i]) : std(c.noisyDist)
              for n in 1:d.N[i]] for i in 1:d.I]

      y = deepcopy(s.y_imputed)

      return DICparam(p, mu, sig, y)
    end
  end


  dden = Matrix{Vector{Float64}}[]

  function update!(s::State, iter::Int, out)
    update_state!(s, c, d, tuners, loglike, fix, use_repulsive,
                  joint_update_Z, sb_ibp)

    if computedden && iter > nburn && (iter - nburn) % thin_dden == 0
      append!(dden,
              [[datadensity(i, j, s, c, d) for i in 1:d.I, j in 1:d.J]])
    end

    if computeLPML && iter > nburn
      # Inverse likelihood for each data point
      like = [[compute_like(i, n, s, c, d) for n in 1:d.N[i]] for i in 1:d.I]

      # Update (or initialize) CPO
      MCMC.updateCPO(cpoStream, vcat(like...))

      # Add to printMsg
      printMsg(iter, " -- LPML: $(MCMC.computeLPML(cpoStream))")
    end

    if computeDIC && iter > nburn
      # Update DIC
      MCMC.updateDIC(dicStream, s, updateParams,
                     loglikeDIC, convertStateToDicParam)

      # Add to printMsg
      printMsg(iter, " -- DIC: $(MCMC.computeDIC(dicStream, loglikeDIC,
                                                 paramMeanCompute))")
    end

    printMsg(iter, "\n")
    if flushOutput
      flush(stdout)
    end
  end

  if isinf(compute_loglike(init, c, d, normalize=true))
    println("Warning: Initial state yields likelihood of zero.")
    msg = """
    It is likely the case that the initialization of missing values
    is not consistent with the provided missing mechanism. The MCMC
    will almost certainly reject the initial values and sample new
    ones in its place.
    """
    println(join(split(msg), " "))
  else
    println("")
  end

  out, lastState = MCMC.gibbs(init, update!, monitors=monitors,
                              thins=thins, nmcmc=nmcmc, nburn=nburn,
                              printFreq=printFreq,
                              loglike=loglike, flushOutput=flushOutput,
                              printlnAfterMsg=!(computeDIC || computeLPML))

  mega_out = out, lastState, loglike

  if computeDIC || computeLPML
    LPML = computeLPML ? MCMC.computeLPML(cpoStream) : NaN
    Dmean, pD = computeDIC ? MCMC.computeDIC(dicStream,
                                             loglikeDIC,
                                             paramMeanCompute,
                                             return_Dmean_pD=true) : (NaN, NaN)
    metrics = Dict(:LPML => LPML,
                   :DIC => Dmean + pD,
                   :Dmean => Dmean,
                   :pD => pD)
    println()
    println("metrics:")
    for (k, v) in metrics
      println("$k => $v")
    end

    mega_out = mega_out ..., metrics
  end

  if computedden
    mega_out = mega_out ..., dden
  end

  return mega_out
end
