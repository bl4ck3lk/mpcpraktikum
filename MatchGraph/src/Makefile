
# Add source files here
TARGET		:= Main
# Cuda source files (compiled with cudacc)
CUFILES		:= GPUMatrix.cu GPUSparse.cu CMEstimatorGPUSparse.cu InitializerGPU.cu Helper.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Main.cpp CPUImpl.cpp GPUComparator.cpp CMEstimatorCPU.cpp Tester.cpp ImageHandler.cpp CPUComparator.cpp InitializerCPU.cpp Comparator_CVGPU.cpp Comparator_CPU.cpp

#Paths to be set
#### CUDA #####
#export CUDA_INSTALL_PATH="/graphics/projects/cuda/toolkit_5.0/cuda/"
#export PATH="./:/graphics/projects/cuda/toolkit_5.0/cuda/open64/bin/:/graphics/projects/cuda/toolkit_5.0/cuda/bin/:/graphics/projects/cuda/toolkit_5.0/cuda/nvvm/:$PATH"
#export LD_LIBRARY_PATH="/graphics/projects/cuda/toolkit_5.0/cuda/lib64/:$LD_LIBRARY_PATH"

#### CULA ####
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/schwarzk/cula_sparse/lib64/
#export CULASPARSE_INC_PATH="/home/schwarzk/cula_sparse/include"
#export CULASPARSE_LIB_PATH_64="/home/schwarzk/cula_sparse/lib64"

#### OPEN CV ####
#export PKG_CONFIG_PATH=/graphics/projects/opt/OpenCV-2.4.5_cuda/lib/pkgconfig
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/graphics/projects/opt/OpenCV-2.4.5_cuda/lib

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

PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/graphics/projects/opt/OpenCV-2.4.5_cuda/lib/pkgconfig
export PKG_CONFIG_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/graphics/projects/opt/OpenCV-2.4.1_cuda/lib
#export LD_LIBRARY_PATH
INCLUDES  += -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/C/common/inc -I /graphics/projects/cuda/sdk_4.2/CUDALibraries/common/inc -I /lib/Eigen -I /graphics/projects/opt/OpenCV-2.4.5_cuda/include/opencv2/nonfree -I /graphics/projects/cuda/cula_r14/include -I /home/schwarzk/cula_sparse/include `pkg-config opencv --cflags` 

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

LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -lcula_sparse -lcublas -liomp5 -lpthread -lcolamd -lcusparse -lcudart -L /graphics/projects/cuda/cula_r14/lib64 -L /home/schwarzk/cula_sparse/lib64 `pkg-config opencv --libs`


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
