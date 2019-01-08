.PHONY: send retrieve retrievelite litecp 
.PHONY: run pullAndProcess postProcess repFamTest test testlocal testplots metrics 
.PHONY: genpreprocessGraphs sendPreprocessedGraphs getPreprocessedGraphs getRData sendRData

# Current run name
DIR_CB_RUN = cb_preprocessed_smartinit_randeps_cauchy

# AWS bucket for preprocessed data
AWS_BUCKET_CB_PREPROCESSED = s3://cytof-cb-results/cb-preprocessed
# Current run dir
RESULTS_DIR_CB_RUN = results/$(DIR_CB_RUN)/
# Current AWS bucket
AWS_BUCKET_CB_RUN = s3://cytof-cb-results/$(subst _,-,$(DIR_CB_RUN))

# {{{1
# DIR_CB_RUN = cb_preprocessed_fixed_quadmiss_smartinit_lpml_rmbadrows_lamzero_cauchy
#
# DIR_CB_RUN = cb_preprocessed_fixed_quadmiss_smartinit_lpml_rmbadrows_lamzero_cauchy_sig2fixed
#
# DIR_CB_RUN = cb_preprocessed_fixed_quadmiss_smartinit_lpml_rmbadrows_lamzero_cauchy_sig2fixed0.1
# RESULTS_DIR_CB_RUN = results/$(DIR_CB_RUN)/
# AWS_BUCKET_CB_RUN = s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit-lpml-rmbadrows-lamzero-cauchy-sig2fixed0.1
#
# DIR_CB_RUN = cb_preprocessed_fixed_quadmiss_smartinit_lpml_rmbadrows_lamzero_cauchy
# RESULTS_DIR_CB_RUN = results/$(DIR_CB_RUN)/
# AWS_BUCKET_CB_RUN = s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit-lpml-rmbadrows-lamzero-cauchy
# # RESULTS_DIR_CB_RUN3 = "results/cb_preprocessed_fixed_b1/"
# AWS_BUCKET_CB_RUN3 = "s3://cytof-cb-results/cb-preprocessed-fixed-b1"
# 
# RESULTS_DIR_CB_RUN4 = "results/cb_preprocessed_fixed_quadmiss/"
# AWS_BUCKET_CB_RUN4="s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss"

# DIR_CB_RUN5 = "cb_preprocessed_fixed_quadmiss_smartinit/"
# RESULTS_DIR_CB_RUN5 = "results/cb_preprocessed_fixed_quadmiss_smartinit/"
# AWS_BUCKET_CB_RUN5="s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit"

# DIR_CB_RUN5 = cb_preprocessed_fixed_quadmiss_smartinit_lpml
# RESULTS_DIR_CB_RUN5 = "results/$(DIR_CB_RUN5)/"
# AWS_BUCKET_CB_RUN5 = "s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit-lpml"

# DIR_CB_RUN5 = cb_preprocessed_fixed_quadmiss_smartinit_lpml_rmbadrows
# RESULTS_DIR_CB_RUN5 = "results/$(DIR_CB_RUN5)/"
# AWS_BUCKET_CB_RUN5 = "s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit-lpml-rmbadrows"

# DIR_CB_RUN5 = cb_preprocessed_fixed_quadmiss_smartinit_lpml_rmbadrows_lamzero
# RESULTS_DIR_CB_RUN5 = "results/$(DIR_CB_RUN5)/"
# AWS_BUCKET_CB_RUN5 = "s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-smartinit-lpml-rmbadrows-lamzero"

# DIR_CB_RUN5 = cb_preprocessed_fixed_quadmiss_lpml_rmbadrows_lamzero
# RESULTS_DIR_CB_RUN5 = "results/$(DIR_CB_RUN5)/"
# AWS_BUCKET_CB_RUN5 = "s3://cytof-cb-results/cb-preprocessed-fixed-quadmiss-lpml-rmbadrows-lamzero"
# }}}1

repFamTest:
	./run $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --repFAM

run:
	./run $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN)

postProcess:
	julia genResults.jl $(RESULTS_DIR_CB_RUN)
	aws s3 sync $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --exclude '*.nfs*'

send:
	aws s3 sync $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --exclude '*.nfs*'

retrieve:
	aws s3 sync $(AWS_BUCKET_CB_RUN) $(RESULTS_DIR_CB_RUN)

test:
	./run $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --test

testlocal:
	./run results/testlocal/ $(AWS_BUCKET_CB_RUN) --test --testlocal

testplots:
	./run results/testplots/ $(AWS_BUCKET_CB_RUN) --test --testplots

litecp:
	rsync -av --exclude '*.jld*' $(RESULTS_DIR_CB_RUN) .sandbox/$(DIR_CB_RUN)

metrics: metrics.py
	python3 metrics.py $(RESULTS_DIR_CB_RUN)

pullAndProcess:
	julia pullAndGenResults.jl $(AWS_BUCKET_CB_RUN)

retrievelite:
	aws s3 sync $(AWS_BUCKET_CB_RUN) $(RESULTS_DIR_CB_RUN) --exclude '*.jld2'


# Preprocess ###
genpreprocessGraphs:
	julia gen_preprocessed_data.jl

sendPreprocessedGraphs:
	aws s3 sync results/misc/img/ s3://cytof-cb-results/misc/img/

getPreprocessedGraphs:
	aws s3 sync s3://cytof-cb-results/misc/img/ results/misc/img/

# get r data
getRData:
	aws s3 sync s3://cytof-cb-data/ data/

# send r data
sendRData:
	aws s3 sync data/ s3://cytof-cb-data/


