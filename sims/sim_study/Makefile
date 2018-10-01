.PHONY: sim


sim:
	 ./sim.sh

postProcess:
	./genResults
	#for d in `ls results/`; do echo $$d; done
	#for d in `ls results/`; do julia post_process.jl $$d; done

zip:
	zip -r .sandbox/results.zip results -x \*.jld2

litecp:
	rsync -av --exclude '*.jld*' results .sandbox/results

sandboxClean:
	rm -rf .sandbox/*

imgClean:
	rm -rf results/*/img/*

clean:
	rm -rf results/*

