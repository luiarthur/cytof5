t: 
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();' --color=yes 

sim: 
	time julia -e 'import Pkg; Pkg.activate("."); cd("test/"); include("sim_test.jl")' --color=yes 


