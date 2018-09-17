# Run in parallel
MAKEFLAGS += -j4

# Simulation variables.
I=3
J=32
K=4
L=4
K_MCMC=10
L_MCMC=5

# Test package.
t: 
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();' --color=yes 

# Simulation
sim: sim100 sim1000 sim10000

sim100:
	n=100; mkdir -p test/sims/result/N$$n/
	n=100; cd test/sims/ && julia sim.jl $(I) $(J) $$n $(K) $(L) $(K_MCMC) $(L_MCMC) > result/N$$n/log.txt

sim1000:
	n=1000; mkdir -p test/sims/result/N$$n/
	n=1000; cd test/sims/ && julia sim.jl $(I) $(J) $$n $(K) $(L) $(K_MCMC) $(L_MCMC) > result/N$$n/log.txt

sim10000:
	n=10000; mkdir -p test/sims/result/N$$n/
	n=10000; cd test/sims/ && julia sim.jl $(I) $(J) $$n $(K) $(L) $(K_MCMC) $(L_MCMC) > result/N$$n/log.txt

clean:
	rm -rf test/sims/result/*
