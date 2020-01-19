module Simulation

using Distributions

function f(id)
  rv = mean(rand(Gamma(2, 3), 100))
  msg = "job: $(id) | rv: (rv)"
  println("pid: $(getpid()) | Threads: $(Threads.nthreads()) | $(msg)")
  sleep(1)
  return 
end

end # module Simulation

