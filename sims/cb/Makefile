.PHONY: send retrieve retrieveall litecp clean zmetric
.PHONY: run pullAndProcess postProcess repFamTest test testlocal testplots metrics runAndProcess missmechsen
.PHONY: genpreprocessGraphs sendPreprocessedGraphs getPreprocessedGraphs getRData sendRData
.PHONY: sendflowsom getflowsom flowsom

# Current run name
DIR_CB_RUN = cb-paper
# Current run dir
# RESULTS_DIR_CB_RUN = results/$(DIR_CB_RUN)/
RESULTS_DIR_CB_RUN = $(SCRATCH_DIR)/cytof/results/$(DIR_CB_RUN)/
# Current AWS bucket
AWS_BUCKET_CB_RUN = s3://cytof-cb-results/$(subst _,-,$(DIR_CB_RUN))

# AWS bucket for preprocessed data
AWS_BUCKET_CB_PREPROCESSED = s3://cytof-cb-results/cb-preprocessed

# {{{1
# NOTE: (for paper)
# DIR_CB_RUN = cb-paper
#
# NOTE: (other runs)
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_imputenear_sweep_fulldata2
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_imputenear_sweep_fulldata
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_imputenear_sweep
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_customimpute
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_imputenear
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_imputefar
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_prior_L2
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_delta_prior
# DIR_CB_RUN = cb_smartinit_randeps_zmarg_iota_noisyremoved
# DIR_CB_RUN = cb_preprocessed_smartinit_randeps_zmarg_iota
# DIR_CB_RUN = cb_preprocessed_smartinit_randeps_zmarg_gap_correctedmu_smallnoisyvar
# DIR_CB_RUN = cb_preprocessed_smartinit_randeps_zmarg_gap_etaunif
# DIR_CB_RUN = cb_preprocessed_smartinit_randeps_zmarg_gap
# DIR_CB_RUN = cb_preprocessed_smartinit_randeps_zjointup
# DIR_CB_RUN = cb_preprocessed_smartinit_randeps
#
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

revisions-multiple-runs:
	mkdir -p $(RESULTS_DIR_CB_RUN)
	bash revisions-multiple-runs.sh  $(RESULTS_DIR_CB_RUN)/  $(AWS_BUCKET_CB_RUN)

repFamTest:
	./run $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --repFAM

run:
	./run $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN)

missmechsen:
	bash missmechsen.sh  $(RESULTS_DIR_CB_RUN)/  $(AWS_BUCKET_CB_RUN)

runAndProcess: run pullAndProcess

postProcess:
	julia genResults.jl $(RESULTS_DIR_CB_RUN)
	aws s3 sync $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --exclude '*.nfs*'

send:
	aws s3 sync $(RESULTS_DIR_CB_RUN) $(AWS_BUCKET_CB_RUN) --exclude '*.nfs*'

getBestCB:
	aws s3 sync $(AWS_BUCKET_CB_RUN)/K_MCMC21_L0_MCMC5_L1_MCMC3_tau0_10.0_tau1_10.0_SEED0_smartinittrue_subsample1.0_dnoisynormal_noisy_scale3.16 $(RESULTS_DIR_CB_RUN)/K_MCMC21_L0_MCMC5_L1_MCMC3_tau0_10.0_tau1_10.0_SEED0_smartinittrue_subsample1.0_dnoisynormal_noisy_scale3.16


retrieveall:
	aws s3 sync $(AWS_BUCKET_CB_RUN) $(RESULTS_DIR_CB_RUN) --exclude '*EXCLUDE_ME*'

retrieve:
	aws s3 sync $(AWS_BUCKET_CB_RUN) $(RESULTS_DIR_CB_RUN) --exclude '*.jld2' --exclude '*EXCLUDE_ME*'

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
	./lpml_vs_small_clus.py

pullAndProcess:
	julia pullAndGenResults.jl $(AWS_BUCKET_CB_RUN)


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

# Send nimble gmm results
sendNimbleGmm:
	aws s3 sync results/gmm/ s3://cytof-cb-results/numble-gmm/

# Send nimble gmm results
getNimbleGmm:
	aws s3 sync s3://cytof-cb-results/numble-gmm/ results/gmm/ 

# Clean results
clean:
	rm -rf $(RESULTS_DIR_CB_RUN)/*

# z_metric
zmetric: metrics
	julia compute_z_metric.jl


# Do flowsom on CB
flowsom:
	julia flowsom.jl

# send FlowSOM CB results
sendflowsom:
	aws s3 sync results/cb-paper/flowsom s3://cytof-cb-results/cb-paper/flowsom/

# send metrics
sendmetrics:
	aws s3 sync $(RESULTS_DIR_CB_RUN)/metrics $(AWS_BUCKET_CB_RUN)/metrics/
