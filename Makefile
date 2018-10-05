.PHONY: test

# Test package.
test: 
	julia -e 'import Pkg; Pkg.pkg"activate ."; Pkg.test();' --color=yes 

install:
	julia -e 'import Pkg; Pkg.pkg"dev .; precompile"; using Cytof5' --color=yes

