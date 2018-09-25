# Run in parallel
MAKEFLAGS += -j4

# Simulation variables.
I=3
J=32
K=4
L=4
K_MCMC=10
L_MCMC=5
NFACTORS := 100 1000 10000
SIMS := $(addprefix sim, $(NFACTORS))
SIMDIR := "sims/sim_study/"

# This will make a sequence from 1 to LAST
#LAST := 4
#NFACTORS := $(shell seq 1 $(LAST))

# Test package.
t: 
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();' --color=yes 

# Simulation.
.PHONY: sims $(SIMS)

sims: $(SIMS)
	echo "Completed simulations."

$(SIMS): sim%:
	$(eval n := $*)
	mkdir -p $(SIMDIR)/result/N$(n)
	cd $(SIMDIR) && julia sim.jl $(I) $(J) $(n) $(K) $(L) $(K_MCMC) $(L_MCMC) > result/N$(n)/log.txt

# Remove artifacts from simulations.
clean:
	rm -rf $(SIMDIR)/result/*

