t: 
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();' --color=yes 

sim100: 
	cd test/sims/ && pwd && julia sim_test.jl 3 32 100 4 4 10 5

sim1000: 
	cd test/sims/ && pwd && julia sim_test.jl 3 32 1000 4 4 10 5

sim10000: 
	cd test/sims/ && pwd && julia sim_test.jl 3 32 10000 4 4 10 5


