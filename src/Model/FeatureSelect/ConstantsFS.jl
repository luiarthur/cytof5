"""
ConstantsFS object for FAM with feature selection.
Augments Constants object with new priors:
W_star_prior, p_prior, and omega_prior.
Note that only one of p_prior or omega_prior will be used. 
"""
mutable struct ConstantsFS{T <: ContinuousDistribution}
  W_star_prior::Gamma  # W*_ik ~ Gamma(a_W, 1)
  p_prior::Beta  # p_c ~ Beta(a_p, b_p)
  omega_prior::T  # omage_each ~ Normal(m_omega , s_omega)
  constants::Constants
end

function ConstantsFS(c::Constants)
  # ws_prior = Gamma(1. / c.K, 1.)  # Old version.
  ws_prior = Gamma(10., 1.)
  p_prior = Beta(.1, .1)
  omega_prior = Normal(-5, 1.)

  return ConstantsFS(ws_prior, p_prior, omega_prior, c)
end


function printConstants(c::ConstantsFS, preprintln::Bool=true)
  if preprintln
    println("ConstantsFS:")
  end

  _fieldnames = collect(fieldnames(typeof(c)))

  for fname in filter(fn -> fn != :constants, _fieldnames)
    x = getfield(c, fname)
    T = typeof(x)
    if T <: Number
      println("$fname: $x")
    elseif T <: Vector
      N = length(x)
      for i in 1:N
        println("$(fname)[$i]: $(x[i])")
      end
    elseif T <: Dict
      for (k, v) in x
        println("$(fname)[$k]: $v")
      end
    else
      println("$fname: $x")
    end
  end

  printConstants(c.constants, true)
end
