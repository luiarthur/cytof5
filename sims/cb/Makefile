AWS_BUCKET_CB_PREPROCESSED = "s3://cytof-cb-results/cb-preprocessed"

RESULTS_DIR_CB_RUN3 = "results/cb_preprocessed_fixed_b1/"
AWS_BUCKET_CB_RUN3 = "s3://cytof-cb-results/cb-preprocessed-fixed-b1"

RESULTS_DIR_CB_RUN4 = "results/cb_preprocessed_fixed_quadmiss/"
AWS_BUCKET_CB_RUN4="s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss"

.PHONY: run test postProcess postProcess2 clean retrieve 
.PHONY: send2 retrieve2 retrieve run3 test3 postProcess3 litecp3 metrics
.PHONY: send4 retrieve4 run4 test3 postProcess4 litecp4 metrics4 pullAndProcess4

run:
	 ./run

postProcess:
	for d in `ls results/`; do echo $$d && julia post_process.jl results/$$d; done

imgClean:
	rm -rf results/*/img/*

clean:
	rm -rf results/*

retrieve:
	aws s3 sync s3://cytof-cb-results/ results/


# Run2 ################################################################################
run2:
	./run

postProcess2:
	julia genResults.jl results/cb_preprocessed/
	aws s3 sync results/cb_preprocessed $(AWS_BUCKET_CB_PREPROCESSED) --exclude '*.nfs*'

send2:
	aws s3 sync results/cb_preprocessed $(AWS_BUCKET_CB_PREPROCESSED) --exclude '*.nfs*'

retrieve2:
	aws s3 sync $(AWS_BUCKET_CB_PREPROCESSED) results/cb_preprocessed

test:
	./run --test

litecp2:
	rsync -av --exclude '*.jld*' results/cb_preprocessed .sandbox/


# Run3 ################################################################################
run3:
	./run3 $(RESULTS_DIR_CB_RUN3) $(AWS_BUCKET_CB_RUN3)

postProcess3:
	julia genResults.jl $(RESULTS_DIR_CB_RUN3)
	aws s3 sync $(RESULTS_DIR_CB_RUN3) $(AWS_BUCKET_CB_RUN3) --exclude '*.nfs*'

send3:
	aws s3 sync $(RESULTS_DIR_CB_RUN3) $(AWS_BUCKET_CB_RUN3) --exclude '*.nfs*'

retrieve3:
	aws s3 sync $(AWS_BUCKET_CB_RUN3) $(RESULTS_DIR_CB_RUN3)

test3:
	./run3 $(RESULTS_DIR_CB_RUN3) $(AWS_BUCKET_CB_RUN3) --test

litecp3:
	rsync -av --exclude '*.jld*' $(RESULTS_DIR_CB_RUN3) .sandbox/cb_preprocessed_fixed_b1

metrics: metrics.py
	python3 metrics.py $(RESULTS_DIR_CB_RUN3)

# Run4 ################################################################################
run4:
	./run3 $(RESULTS_DIR_CB_RUN4) $(AWS_BUCKET_CB_RUN4)

repFamTest:
	./run3 $(RESULTS_DIR_CB_RUN4) $(AWS_BUCKET_CB_RUN4) --repFAM

postProcess4:
	julia genResults.jl $(RESULTS_DIR_CB_RUN4)
	aws s3 sync $(RESULTS_DIR_CB_RUN4) $(AWS_BUCKET_CB_RUN4) --exclude '*.nfs*'

send4:
	aws s3 sync $(RESULTS_DIR_CB_RUN4) $(AWS_BUCKET_CB_RUN4) --exclude '*.nfs*'

retrieve4:
	aws s3 sync $(AWS_BUCKET_CB_RUN4) $(RESULTS_DIR_CB_RUN4)

test4:
	./run3 $(RESULTS_DIR_CB_RUN4) $(AWS_BUCKET_CB_RUN4) --test

testlocal:
	./run3 results/testlocal/ $(AWS_BUCKET_CB_RUN4) --test --testlocal

litecp4:
	rsync -av --exclude '*.jld*' $(RESULTS_DIR_CB_RUN4) .sandbox/cb_preprocessed_fixed_b1_smallTau

metrics4: metrics.py
	python3 metrics.py $(RESULTS_DIR_CB_RUN4) '\w+repFAMTest'

# Preprocess ###
genpreprocessGraphs:
	julia gen_preprocessed_data.jl

sendPreprocessedGraphs:
	aws s3 sync results/misc/img/ s3://cytof-cb-results/misc/img/

getPreprocessedGraphs:
	aws s3 sync s3://cytof-cb-results/misc/img/ results/misc/img/

# get r data
getRData:
	aws s3 sync s3://cytof-cb-data/ data/ --exclude '*' --include 'cytof_cb.rds'

# send r data
sendRData:
	aws s3 sync data/ s3://cytof-cb-data/ --exclude '*' --include 'cytof_cb.rds'

# pullAndProcess4
pullAndProcess4:
	julia pullAndGenResults.jl $(AWS_BUCKET_CB_RUN4)
