all: 
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();' --color=yes 

