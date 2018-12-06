AWS_BUCKET_CB_PREPROCESSED = "s3://cytof-cb-results/cb-preprocessed"

RESULTS_DIR_CB_RUN3 = "results/cb_preprocessed_fixed_b1/"
AWS_BUCKET_CB_RUN3 = "s3://cytof-cb-results/cb-preprocessed-fixed-b1"

RESULTS_DIR_CB_RUN4 = "results/cb_preprocessed_fixed_quadmiss/"
AWS_BUCKET_CB_RUN4="s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss"

# DIR_CB_RUN5 = "cb_preprocessed_fixed_quadmiss_smartinit/"
# RESULTS_DIR_CB_RUN5 = "results/cb_preprocessed_fixed_quadmiss_smartinit/"
# AWS_BUCKET_CB_RUN5="s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit"

DIR_CB_RUN5 = "cb_preprocessed_fixed_quadmiss_smartinit_lpml/"
RESULTS_DIR_CB_RUN5 = "results/$(DIR_CB_RUN5)/"
AWS_BUCKET_CB_RUN5 = "s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit-lpml"

.PHONY: run test postProcess postProcess2 clean retrieve 
.PHONY: send2 retrieve2 retrieve run3 test3 postProcess3 litecp3 metrics
.PHONY: send4 retrieve4 run4 test4 postProcess4 litecp4 metrics4 pullAndProcess4 retrieve4lite
.PHONY: send5 retrieve5 run5 test5 postProcess5 litecp5 metrics5 pullAndProcess5 retrieve5lite

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
	rsync -av --exclude '*.jld*' $(RESULTS_DIR_CB_RUN4) .sandbox/cb_preprocessed_fixed_quadmiss

metrics4: metrics.py
	python3 metrics.py $(RESULTS_DIR_CB_RUN4)

pullAndProcess4:
	julia pullAndGenResults.jl $(AWS_BUCKET_CB_RUN4)

retrieve4lite:
	aws s3 sync $(AWS_BUCKET_CB_RUN4) $(RESULTS_DIR_CB_RUN4) --exclude '*.jld2'

# Run5 ################################################################################
run5:
	./run5 $(RESULTS_DIR_CB_RUN5) $(AWS_BUCKET_CB_RUN5)

postProcess5:
	julia genResults.jl $(RESULTS_DIR_CB_RUN5)
	aws s3 sync $(RESULTS_DIR_CB_RUN5) $(AWS_BUCKET_CB_RUN5) --exclude '*.nfs*'

send5:
	aws s3 sync $(RESULTS_DIR_CB_RUN5) $(AWS_BUCKET_CB_RUN5) --exclude '*.nfs*'

retrieve5:
	aws s3 sync $(AWS_BUCKET_CB_RUN5) $(RESULTS_DIR_CB_RUN5)

test5:
	./run5 $(RESULTS_DIR_CB_RUN5) $(AWS_BUCKET_CB_RUN5) --test

testlocal5:
	./run5 results/testlocal/ $(AWS_BUCKET_CB_RUN5) --test --testlocal

litecp5:
	rsync -av --exclude '*.jld*' $(RESULTS_DIR_CB_RUN5) .sandbox/$(DIR_CB_RUN5)

metrics5: metrics.py
	python3 metrics.py $(RESULTS_DIR_CB_RUN5)

pullAndProcess5:
	julia pullAndGenResults.jl $(AWS_BUCKET_CB_RUN5)

retrieve5lite:
	aws s3 sync $(AWS_BUCKET_CB_RUN5) $(RESULTS_DIR_CB_RUN5) --exclude '*.jld2'


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


