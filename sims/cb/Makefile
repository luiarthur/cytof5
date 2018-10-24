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
	echo "TO DO!"

send2:
	echo "TO DO!"

retrieve2:
	echo "TO DO!"

test:
	./run --test
