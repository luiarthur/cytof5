AWS_BUCKET_CB_PREPROCESSED = "s3://cytof-cb-results/cb-preprocessed"

.PHONY: run test postProcess postProcess2 clean retrieve send2 retrieve2 retrieve

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


#########################
postProcess2:
	julia genResults.jl results/cb_preprocessed/
	aws s3 sync results/cb_preprocessed $(AWS_BUCKET_CB_PREPROCESSED) --exclude '*.nfs*'

send2:
	aws s3 sync results/cb_preprocessed $(AWS_BUCKET_CB_PREPROCESSED) --exclude '*.nfs*'

retrieve2:
	aws s3 sync $(AWS_BUCKET_CB_PREPROCESSED) results/cb_preprocessed

test:
	./run --test
