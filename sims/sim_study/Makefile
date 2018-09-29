.PHONY: sim


sim:
	 ./sim.sh


postProcess:
	for d in `ls results/`; do echo $$d; done
	#for d in `ls results/`; do julia post_process.jl $$d; done

imgClean:
	rm -rf results/*/img/*


clean:
	rm -rf results/*

