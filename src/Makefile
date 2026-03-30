.PHONY: all clean

SOURCES := $(wildcard *.c)
INCLUDES := $(wildcard *.h)

BINDIR := ../bin
M := $(BINDIR)/main
MS := $(BINDIR)/main-simd
MO := $(BINDIR)/main-o3

# 这里的 -fopenmp 是开启多线程的关键开关
# -lpthread 有时在某些 Linux 发行版上也是必要的，加上更稳妥
CFLAGS_OMP := -fopenmp

all: $(BINDIR) $(M) $(MS) $(MO)

# 确保 bin 目录存在
$(BINDIR):
	mkdir -p $(BINDIR)

$(M): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O0 $(SOURCES) $(CFLAGS_OMP)

$(MS): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O0 -DSIMD $(SOURCES) $(CFLAGS_OMP)

$(MO): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O3 -DSIMD $(SOURCES) $(CFLAGS_OMP)

clean:
	rm -f $(M) $(MS) $(MO)

include MakefileTasks