SIM4_BUCKET = "s3://cytof-vary-kmcmc-n1000/"
.PHONY: sim

dirsize3:
	aws s3 ls --summarize --human-readable --recursive s3://cytof-sim-beta-tuner-init-results/

sim:
	 ./sim.sh

sim4:
	./sim4.sh

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
	aws s3 sync results/sim3/ $(SIM4_BUCKET) --exclude "*.nfs*"

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
	aws s3 sync results/sim3/ $(SIM4_BUCKET) --exclude "*.nfs*"

