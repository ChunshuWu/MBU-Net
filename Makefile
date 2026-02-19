# ---------------------------------------------------------------------------
# File: Makefile
# ---------------------------------------------------------------------------
# Chunshu Wu, Pacific Northwest National Laboratory (PNNL), U.S.
# Richland, 99352, WA, USA. Feb-03-2026.
# ---------------------------------------------------------------------------


NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w -arch=sm_75 -maxrregcount=255 -rdc=true -dlto --use_fast_math -I/usr/include $(EXTRA_CFLAGS)

# CUDA library paths - prioritize cluster CUDA over conda
ifeq ($(CUDA_HOME),)
    # Try common cluster CUDA locations
    ifneq ($(wildcard /usr/local/cuda),)
        CUDA_HOME = /usr/local/cuda
    else ifneq ($(wildcard /opt/cuda),)
        CUDA_HOME = /opt/cuda
    endif
endif

# Set library paths
ifdef CUDA_HOME
    CUDA_LIB_PATH = $(CUDA_HOME)/lib64
    CUDA_STUB_PATH = $(CUDA_HOME)/lib64/stubs
    LIBS = -L$(CUDA_LIB_PATH) -L$(CUDA_STUB_PATH) -lcudart $(CUDA_LIB_PATH)/libcudadevrt.a -lnvidia-ml -lrt -lpthread -ldl
else
    # Fallback to conda (will likely fail for cooperative kernels)
    CUDA_LIB_PATH = $(CONDA_PREFIX)/lib
    LIBS = -L$(CUDA_LIB_PATH) -lcudart -lnvidia-ml -lrt -lpthread -ldl
endif

all: unet

unet: unet.cu param.h kernel.cuh utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ unet.cu $(LIBS)

clean:
	rm -f unet

