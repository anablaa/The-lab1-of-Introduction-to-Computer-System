BINDIR = ../bin

$(BINDIR)/gen: gen.c
	gcc -o $@ gen.c

clean:
	rm $(BINDIR)/gen

prep:
	mkdir -p ../bin

$(BINDIR)/test.txt:
	$(BINDIR)/gen $(BINDIR)/test.txt 4 4 4

$(BINDIR)/1.txt:
	$(BINDIR)/gen $(BINDIR)/1.txt 1024 1024 1024

$(BINDIR)/2.txt:
	$(BINDIR)/gen $(BINDIR)/2.txt 4 32768 4

$(BINDIR)/3.txt:
	$(BINDIR)/gen $(BINDIR)/3.txt 32 32768 32

$(BINDIR)/4.txt:
	$(BINDIR)/gen $(BINDIR)/4.txt 256 32768 256

$(BINDIR)/5.txt:
	$(BINDIR)/gen $(BINDIR)/5.txt 1024 256 256

$(BINDIR)/6.txt:
	$(BINDIR)/gen $(BINDIR)/6.txt 256 256 1024

$(BINDIR)/7.txt:
	$(BINDIR)/gen $(BINDIR)/7.txt 1024 512 512

$(BINDIR)/8.txt:
	$(BINDIR)/gen $(BINDIR)/8.txt 512 512 1024

txt: $(BINDIR)/test.txt $(BINDIR)/1.txt $(BINDIR)/2.txt $(BINDIR)/3.txt $(BINDIR)/4.txt $(BINDIR)/5.txt $(BINDIR)/6.txt $(BINDIR)/7.txt $(BINDIR)/8.txt

all: $(BINDIR)/gen

.PHONY: clean prep txt all
