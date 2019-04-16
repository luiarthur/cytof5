.PHONY: zip genResults litecp sendData getData genData

# AWS BUCKET for simulated data
SIMDATA_AWS_BUCKET = s3://cytof-sim-results/simdata/

# DATA DIR
SIMDATA_DIR = data/

zip:
	zip -r results_flowsom.zip results
	cp -r results .tmp/results_flowsom

genResults:
	./genResults ../sim_study/results/

genResults2:
	./genResults ../sim_study/results/sim2/

litecp:
	rsync -av --exclude '*.jld*' --exclude '*.nfs*' results/flowSearch/ .tmp/results_flowsom/

sendData:
	aws s3 sync $(SIMDATA_DIR) $(SIMDATA_AWS_BUCKET)

getData:
	aws s3 sync $(SIMDATA_AWS_BUCKET) $(SIMDATA_DIR)

genData:
	julia findBadFlowSOM.jl
