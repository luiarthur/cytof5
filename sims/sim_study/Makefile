SIM4_BUCKET = "s3://cytof-vary-kmcmc-n1000/"
SIM5_BUCKET = "s3://cytof-vary-kmcmc-n10000/"
.PHONY: sim simtest

dirsize3:
	aws s3 ls --summarize --human-readable --recursive s3://cytof-sim-beta-tuner-init-results/

sim:
	 ./sim.sh

sim4:
	./sim4.sh

sim5:
	./sim5.sh

postProcess:
	./genResults.py results/
	#for d in `ls results/`; do echo $$d; done
	#for d in `ls results/`; do julia post_process.jl $$d; done

postProcess2:
	./parseMetrics.py results/sim2/
	./genResults.py results/sim2/
	./combineARI results/sim2/

postProcess3:
	./genResults.jl results/sim3/

postProcess4:
	./genResults.jl results/sim4/
	aws s3 sync results/sim4/ $(SIM4_BUCKET) --exclude "*.nfs*"

postProcess5:
	./genResults.jl results/sim5/
	aws s3 sync results/sim5/ $(SIM5_BUCKET) --exclude "*.nfs*"


parseMetrics4:
	./parseMetrics4.py results/sim4/

parseMetrics5:
	./parseMetrics4.py results/sim5/

zip:
	zip -r .sandbox/results.zip results -x \*.jld2

zip2:
	zip -r .sandbox/results_varying_K.zip results/sim2/ -x \*.jld2


litecp:
	rsync -av --exclude '*.jld*' results .sandbox/results

litecp2:
	rsync -av --exclude '*.jld*' results/sim2/ .sandbox/results/sim2

litecp3:
	rsync -av --exclude '*.jld*' --exclude '*.nfs*' results/sim3/ .sandbox/results/sim3

litecp4:
	mkdir -p .sandbox/results/sim4
	rsync -av --exclude '*.jld*' --exclude '*.nfs*' results/sim4/ .sandbox/results/sim4

litecp5:
	mkdir -p .sandbox/results/sim5
	rsync -av --exclude '*.jld*' --exclude '*.nfs*' results/sim5/ .sandbox/results/sim5


sandboxClean:
	rm -rf .sandbox/*

imgClean:
	rm -rf results/*/img/*

clean:
	rm -rf results/*


# Simulation 1 AWS to and fro
retrieve:
	aws s3 sync s3://cytof-sim-results/ results/

send:
	aws s3 sync results/ s3://cytof-sim-results/

# Simulation 2 AWS to and fro
retrieve2:
	aws s3 sync s3://cytof-sim2-results/ results/sim2/

send2:
	aws s3 sync results/sim2/ s3://cytof-sim2-results/

# Simulation 3 AWS to and fro
retrieve3:
	aws s3 sync s3://cytof-sim-beta-tuner-init-results/ results/sim3/ --exclude "*.nfs*"
send3:
	aws s3 sync results/sim3/ s3://cytof-sim-beta-tuner-init-results/ --exclude "*.nfs*"

# Simulation 4 AWS to and fro
retrieve4:
	aws s3 sync $(SIM4_BUCKET) results/sim4/ --exclude "*.nfs*"
send4:
	aws s3 sync results/sim4/ $(SIM4_BUCKET) --exclude "*.nfs*"


# Simulation 5 AWS to and fro
retrieve5:
	aws s3 sync $(SIM5_BUCKET) results/sim5/ --exclude "*.nfs*"
send5:
	aws s3 sync results/sim5/ $(SIM5_BUCKET) --exclude "*.nfs*"

simtest:
	julia sim.jl --I=3 --J=32 --N_factor=100 --K=8 --K_MCMC=10 \
		--L0=4 --L1=4 --L0_MCMC=5 --L1_MCMC=5 \
	  --b0PriorSd=0.3 --b1PriorScale=0.3 \
		--RESULTS_DIR=results/test/ \
		--EXP_NAME=bla --MCMC_ITER=1000 --BURN=10000 \
		--b0TunerInit=0.1 --b1TunerInit=0.1 \
		--fix_b1=true --printFreq=50
	./genResults.jl results/test/
	rm -f result/test/bla/*.jld2

simtestlocal:
	julia sim.jl --I=3 --J=32 --N_factor=100 --K=8 --K_MCMC=10 \
		--L0=4 --L1=4 --L0_MCMC=5 --L1_MCMC=5 \
	  --b0PriorSd=0.3 --b1PriorScale=0.3 \
		--RESULTS_DIR=results/test/ \
		--EXP_NAME=bla --MCMC_ITER=1000 --BURN=10000 \
		--b0TunerInit=0.1 --b1TunerInit=0.1 \
		--fix_b1=true --printFreq=50
	./genResults.jl results/test/
