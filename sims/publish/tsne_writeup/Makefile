SHELL := /bin/bash

.PHONY: all build clean getimg

FNAME = tsne
CB_AWS_BUCKET = s3://cytof-cb-results/cb-paper
AWS_TSNE_IMG_DIR = $(CB_AWS_BUCKET)/K_MCMC21_L0_MCMC5_L1_MCMC3_tau0_10.0_tau1_10.0_SEED0_smartinittrue_subsample1.0_dnoisynormal_noisy_scale3.16/img/tsne/init_neg6

all: build clean

getimg:
	aws s3 cp $(AWS_TSNE_IMG_DIR)/tsne_sample_1.pdf img/
	aws s3 cp $(AWS_TSNE_IMG_DIR)/tsne_sample_2.pdf img/
	aws s3 cp $(AWS_TSNE_IMG_DIR)/tsne_sample_3.pdf img/

build:
	pdflatex $(FNAME).tex

clean:
	rm -f *.{log,aux,nav,out,snm,toc,blg,bbl,fdb_latexmk,fls,synctex.gz,zip,pdfsync}


