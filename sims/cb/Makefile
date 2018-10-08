.PHONY: run

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

