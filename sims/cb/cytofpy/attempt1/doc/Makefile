.PHONY: clean preview all githubify

FNAME = advi_cytof

all: preview

preview:
	R --slave -e "rmarkdown::render('$(FNAME).Rmd', rmarkdown::pdf_document(toc=TRUE))"
	# R --slave -e "rmarkdown::render('$(FNAME).Rmd')"
	# mv $(FNAME).md README.md

clean:
	rm -f *.html *.md *.pdf
	rm -rf $(FNAME)_files

githubify:
	time R --slave -e "rmarkdown::render('$(FNAME).Rmd', rmarkdown::github_document(pandoc_args='--webtex'))"
	rm -f *.html
	mv $(FNAME).md README.md

