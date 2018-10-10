.PHONY: sim


sim:
	 ./sim.sh

postProcess:
	./genResults results/
	#for d in `ls results/`; do echo $$d; done
	#for d in `ls results/`; do julia post_process.jl $$d; done

postProcess2:
	./parseMetrics results/sim2/
	./genResults results/sim2/

zip:
	zip -r .sandbox/results.zip results -x \*.jld2

zip2:
	zip -r .sandbox/results_varying_K.zip results/sim2/ -x \*.jld2


litecp:
	rsync -av --exclude '*.jld*' results .sandbox/results

sandboxClean:
	rm -rf .sandbox/*

imgClean:
	rm -rf results/*/img/*

clean:
	rm -rf results/*

# Simulation 2 AWS to and fro
retrieve2:
	aws s3 sync s3://cytof-sim2-results/ results/sim2/

send2:
	aws s3 sync results/sim2/ s3://cytof-sim2-results/

# Simulation 1 AWS to and fro
retrieve:
	aws s3 sync s3://cytof-sim-results/ results/

send:
	aws s3 sync results/ s3://cytof-sim-results/
