.PHONY: prep txt compile grade clean log plot

prep:
	make -C utils prep
	make -C utils

txt:
	make -C utils txt

compile:
	make -C src

grade:
	make -C src task2
	make -C src task3

clean:
	make -C src clean
	make -C utils clean

Is:=1 2 3 4 5 6 7 8

log1:
	@if [ -e "tasks1_run.log" ]; then rm tasks1_run.log; fi
	$(foreach item, $(Is), \
		make -C src tasks1 I=$(item) | grep Machine | awk '{print $5}' >> tasks1_run.log ; echo '' >> tasks1_run.log ;\
		)

log2:
	@if [ -e "tasks2_run.log" ]; then rm tasks2_run.log; fi
	$(foreach item, $(Is), \
		make -C src tasks2 I=$(item) | grep Machine | awk '{print $5}' >> tasks2_run.log ; echo '' >> tasks2_run.log ;\
		)

log3:
	@if [ -e "tasks3_run.log" ]; then rm tasks3_run.log; fi
	$(foreach item, $(Is), \
		make -C src tasks3 I=$(item) | grep Machine | awk '{print $5}' >> tasks3_run.log ; echo '' >> tasks3_run.log ;\
		)

log: log1 log2 log3

plot:
	python utils/plot.py

test:
	@if [ -e "z.log" ]; then rm -f z.log; fi
	make -C src mkn I=test