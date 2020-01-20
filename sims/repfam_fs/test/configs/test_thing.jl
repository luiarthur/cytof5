module Simulation

using Distributions

function redirect_all(f::Function, fout::String, ferr::String)
  open(fout, "w") do ioout
    open(ferr, "w") do ioerr
      redirect_stdout(ioout) do
        redirect_stderr(ioerr) do
          f()
        end
      end
    end
  end
end

function redirect_all(f::Function, path::String)
  redirect_all("$(path).out", "$(path).err") do
    f()
  end
end

function f(id)
  redirect_all("configs/out-$(id)") do
    rv = mean(rand(Gamma(2, 3), 100))
    msg = "job: $(id) | rv: $(rv)"
    println("pid: $(getpid()) | Threads: $(Threads.nthreads()) | $(msg)")
    x = randn() > 0 ? 1 : z
    @time sleep(x); flush(stdout)
    @time sleep(x); flush(stdout)
  end
end

end # module Simulation
