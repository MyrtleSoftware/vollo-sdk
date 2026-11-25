all: vollo-gpu-input

NVCC ?= nvcc
# Machine code for every major GPU architecture, so the driver never has to compile the kernel's
# PTX, which fails when the toolkit is newer than the driver. `make CUDA_ARCH=native` builds for
# this machine's GPUs only.
CUDA_ARCH ?= all-major
NVCCFLAGS += -O2 -std=c++17 -arch=$(CUDA_ARCH) -I$(VOLLO_SDK)/include
LDFLAGS += -L$(VOLLO_SDK)/lib
LDLIBS += -lvollo_rt

.PHONY: all clean

vollo-gpu-input: gpu_input.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f vollo-gpu-input
