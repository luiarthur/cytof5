module Util

"""
create annotations for any struct
"""
macro namedargs(StructDec)
  sstruct = string(StructDec)
  typename = sstruct[findfirst(r"(?<=struct).*(?=\n)", sstruct)]
  typename = strip(typename)
  parametric = findfirst(r"{.*}", typename)
  if parametric == nothing
    parametric = ""
  else
    parametric = "where $(typename[parametric])"
  end
  typename = typename[findfirst(r"\w+", typename)]

  function fieldNameAndType(expr)
    return split(string(expr), "::")
  end

  args = StructDec.args[3].args[2:2:end]
  # args = filter(a -> !occursin(r"function\s+\w+", string(a)), args)
  args = filter(a -> !occursin(r"new\(+", string(a)), args)
  args = map(a -> fieldNameAndType(a), args)
  fnames = [a[1] for a in args]
  ftypes = [a[2] for a in args]

  callArgs = join(["$(a[1])" for a in args], ", ")
  oldConstructorCall = "$typename($callArgs)"

  fn_args = join(["$(a[1])::$(a[2])" for a in args], ", ")
  namedArgsConstructor = "$typename(; $fn_args) $parametric"

  return quote
    $(esc(StructDec))
    $(esc(Meta.parse("$namedArgsConstructor = $oldConstructorCall")))
  end
end

end # module Util

#= Test. TODO: Put in runtests.jl
import .Util.@namedargs

@namedargs struct Bab{T <: Number}
  x::Int
  y::T
end

@namedargs struct Bob
  x::Int
  y::String
end
beb = Bob(x=1, y="sos")

@namedargs struct Beb{S <: Number, T <: Int}
  x::S
  y::T
end
beb = Beb(x=1.0, y=2)
beb.x += 1 # should not work (immutable)

# Note that the first version is slower here.
@time for i in 1:100000000
  beb = Beb(x=1.0, y=2)
end

@time for i in 1:100000000
  beb = Beb(1.0, 2)
end

# But the timings here are the same
@time [Beb(x=1.0, y=2) for i in 1:10000];
@time [Beb(1.0, 2) for i in 1:10000];


@namedargs mutable struct Bib{S <: Number, T <: Int}
  x::S
  y::T
end
bib = Bib(x=1, y=2.0)
bib.x += 1 # this should work

@namedargs struct Bub{T <: Number}
  x::Int
  y::T
end;

=#
