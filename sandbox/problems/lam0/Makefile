.PHONY: clean build all zip

FNAME = lam0

all: build

build: $(FNAME).pdf

$(FNAME).pdf: $(FNAME).Rmd
	R --slave -e "rmarkdown::render('$(FNAME).Rmd')"

clean:
	rm -f $(FNAME).pdf
	rm -f *.zip
	
zip: build
	zip -r $(FNAME).zip ../ -x '*.swp' '*.gitignore'
