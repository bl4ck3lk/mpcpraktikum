# Uncomment for Debug built
#dbg = 1

# Add source files here
TARGET		:= Comparator_CVGPU
# Cuda source files (compiled with cudacc)
CUFILES		:= ratio.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Comparator_CVGPU.cpp


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

NVCCFLAGS += -arch sm_20


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
#PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/graphics/projects/opt/OpenCV-2.4.1_cuda/lib/pkgconfig
#export PKG_CONFIG_PATH
INCLUDES  += -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/C/common/inc `pkg-config opencv --cflags`
CXXFLAGS += $(INCLUDES)
CFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES)


ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG -deviceemu
else
	COMMONFLAGS += -O3
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
endif

CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(COMMONFLAGS)


LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_DIR)/C/lib -L$(CUDA_SDK_DIR)/C/common/lib/$(OSLOWER) -lcudart `pkg-config opencv --libs`


OBJS := $(patsubst %.cpp,%.o,$(CCFILES))
OBJS += $(patsubst %.c,%.o,$(CFILES))
OBJS += $(patsubst %.cu,%.o,$(CUFILES))

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
	$(NVCC) $(NVCCFLAGS) -o $@.ptx -ptx $<

$(TARGET) : $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
