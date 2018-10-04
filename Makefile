.PHONY: test

# Test package.
test: 
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();' --color=yes 

install:
	julia -e 'import Pkg; Pkg.add("."); using Cytof5'

