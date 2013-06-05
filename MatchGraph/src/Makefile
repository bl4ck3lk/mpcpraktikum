
# Add source files here
TARGET		:= Main
# Cuda source files (compiled with cudacc)
CUFILES		:= CMEstimatorGPUSorted.cu GPUMatrix.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Main.cpp CPUImpl.cpp CPUComparator.cpp CMEstimatorCPU.cpp ImageHandler.cpp Tester.cpp

#/graphics/projects/cuda/cula_r14/lib64

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CXXFLAGS += $(CXXWARN_FLAGS)

CFLAGS += $(CWARN_FLAGS)

NVCCFLAGS += -arch sm_11


################################################################################
# Rules and targets

.SUFFIXES : .cu 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/C/common/inc -I /lib/Eigen -I /graphics/projects/cuda/cula_r14/include
CXXFLAGS += $(INCLUDES)
CFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES)


ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG 
else
	COMMONFLAGS += -O3
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
endif

ifeq ($(emu),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG -deviceemu
endif 


CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(COMMONFLAGS)


LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -L /graphics/projects/cuda/cula_r14/lib64


OBJS := $(patsubst %.cpp,%.o,$(CCFILES))
OBJS += $(patsubst %.c,%.o,$(CFILES))
OBJS += $(patsubst %.cu,%.o,$(CUFILES))

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
#	$(NVCC) $(NVCCFLAGS) -o $@.ptx -ptx $<

$(TARGET) : $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
