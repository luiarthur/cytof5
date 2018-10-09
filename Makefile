.PHONY: test install

# Test package.
test:
	julia -e 'import Pkg; Pkg.pkg"activate ."; Pkg.test();' --color=yes 

# Install Cytof5 from this directory
install:
	julia -e 'import Pkg; Pkg.pkg"dev .; precompile"; using Cytof5' --color=yes

