# Build Instructions

See [here][1] for details.

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```

Alternatively, we can omit the `-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch`
flag if we export the variable name `Torch_DIR=/absolute/path/to/libtorch` in
`.bashrc`. The statement `find_package(TORCH Required)` means look for `Torch_DIR`
in command-line arguments or in environment vars.

[1]: https://pytorch.org/cppdocs/installing.html#minimal-example

