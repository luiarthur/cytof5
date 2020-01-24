# cytof5
Cytof Model Implementation 5

## Virtual environment
The examples in this package should be run in the same environment as the
package. The package environment can be emulated via a virtual environment 
(like in python). This is done in julia as follows:

```julia
import Pkg
Pkg.activate("path/to/Cytof5/")  # activate package
Pkg.instantiate()  # start virtual env
# Pkg.build()  # may be needed to build dependencies
```

## System requirements
- [julia v1.0.0][3]


## New Features in Julia v1.0.0

- Saving and loading data using the [JLD2][1] package
- Linting? [Lint.jl][2]. Not currently up-to-date.



[1]: https://github.com/simonster/JLD2.jl
[2]: https://github.com/tonyhffong/Lint.jl
[3]: https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.0-linux-x86_64.tar.gz
