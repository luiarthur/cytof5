module Util

s3sync(from, to) = run(`aws s3 sync $(from) $(to)`)
s3sync(from, to, tags) = run(`aws s3 sync $(from) $(to) $(tags)`)

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


"""
example usage:

```julia
redirect_all("path/to/out") do
  dosomething()
end
```
"""
function redirect_all(f::Function, path::String)
  redirect_all("$(path).out", "$(path).err") do
    f()
  end
end

end
