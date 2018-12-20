.PHONY: test install issues

# Test package.
test:
	julia -e 'import Pkg; Pkg.pkg"activate ."; Pkg.test();' --color=yes 

# Test package with parallel implementation.
testParallel:
	export JULIA_NUM_THREADS=4; julia -e 'import Pkg; Pkg.pkg"activate ."; Pkg.test();' --color=yes 

# Install Cytof5 from this directory
install:
	julia -e 'import Pkg; Pkg.pkg"dev .; precompile"; using Cytof5' --color=yes

# View Issues
issues:
	hub browse -- issues
