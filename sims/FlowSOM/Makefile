.PHONY: zip genResults

zip:
	zip -r results_flowsom.zip results
	cp -r results .tmp/results_flowsom

genResults:
	./genResults ../sim_study/results/

genResults2:
	./genResults ../sim_study/results/sim2/

litecp:
	rsync -av --exclude '*.jld*' --exclude '*.nfs*' results/flowSearch/ .tmp/results_flowsom/
