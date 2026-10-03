PYTHON ?= python3
PROJECT_ROOT ?= ..
BUILD_DIR ?= build
RVV_KERNEL ?= igemm_kernel_8x4_zvl256b_lmulmf8_unroll2
IME_KERNEL ?= ime_kernel_8x4_zvl256b_lmul1_unroll1
.PHONY: all host-test test
all:
	$(PYTHON) build.py --project-root "$(PROJECT_ROOT)" --output "$(BUILD_DIR)" --cc "$(CC)" --rvv-kernel "$(RVV_KERNEL)" --ime-kernel "$(IME_KERNEL)"
host-test:
	$(PYTHON) build.py --project-root "$(PROJECT_ROOT)" --output "$(BUILD_DIR)" --cc "$(CC)" --host-test
test:
	$(PYTHON) -m unittest discover -s tests -v
