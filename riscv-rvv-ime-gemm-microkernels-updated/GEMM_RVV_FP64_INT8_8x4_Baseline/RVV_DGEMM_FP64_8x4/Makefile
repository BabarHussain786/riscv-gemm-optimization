CC ?= gcc
SUBDIRS := $(sort $(dir $(wildcard dgemm_kernel_8x4_*_*/Makefile)))

.PHONY: all clean

all:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d || exit $$?; done

clean:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d clean || exit $$?; done
