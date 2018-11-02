AWS_BUCKET_CB_PREPROCESSED = "s3://cytof-cb-results/cb-preprocessed"

RESULTS_DIR_CB_RUN3 = "results/cb_preprocessed_fixed_b1/"
AWS_BUCKET_CB_RUN3 = "s3://cytof-cb-results/cb-preprocessed-fixed-b1"

.PHONY: run test postProcess postProcess2 clean retrieve send2 retrieve2 retrieve run3 test3 postProcess3 litecp3

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
	rsync -av --exclude '*.jld*' $(RESULTS_DIR_CB_RUN3) .sandbox/
