SHELL := /bin/bash

.PHONY: build clean all

filename = repFAM

all: build clean

build: $(filename).tex
	pdflatex $(filename).tex
	bibtex $(filename).aux
	pdflatex $(filename).tex
	pdflatex $(filename).tex

clean:
	rm -f *.{log,aux,nav,out,snm,toc,blg,bbl,fdb_latexmk,fls,synctex.gz}
