# MatchGraphCUDA

MatchGraphCUDA uses the Nvidia CUDA GPU Framework to find and retrieve similar elements in a large databases. This implementation is based on the work by [Kim and colleagues](http://www.mpi-inf.mpg.de/~kkim/mgc/MainPaper.pdf).

## Installation and Configuration
This project comes with a pre-configured Makefile, in which your specific username has to be set for the _USER_ variable. Furthermore you have to install the following libraries:

### CUDA
To work with Nvidia CUDA you must have a CUDA compatible GPU device with a properly configured CUDA Toolkit. For more information visit the [CUDA-Zone](https://developer.nvidia.com/category/zone/cuda-zone). 

### Eigen
The CPU version of this project uses the [Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page) for lineare algebra tasks. This is already included here and configured in the Makefile, so there is no need to do anything in this particular case.

### CULA Sparse S5
The GPU version uses CULA Sparse S5 for the linear algebra. For this, you need to download CULA Sparse from [http://www.culatools.com/sparse/](http://www.culatools.com/sparse/), which is free for personal academic usage.  
To compile the program, some environment variables have to be set (respective to the installation path of CULA Sparse):

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$(USER)/cula_sparse/lib64/
    export CULASPARSE_INC_PATH="/home/$(USER)/cula_sparse/include"
    export CULASPARSE_LIB_PATH_64="/home/$(USER)/cula_sparse/lib64
    
As well as some parameters for the compiler, which are already pre-configured in the Makefile. For more details see the [CULA Sparse reference](http://www.culatools.com/cula_sparse_programmers_guide/).

### OpenCV
This implementation was developed and tested with OpenCV 2.4.5

Add this line to your application's Gemfile:

    gem 'sauce_whisk'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install sauce_whisk

## Usage
    
### Running the Program
The program can be run in two different modes for both, the CPU and GPU version. Switching between them can be done by setting the _GPU_VERSION_ variable in the _Main.cpp_ (1 = GPU version, 0 = CPU version).

#### 1. Standard execution:
```bash
    Main <path> <ext> <iter> [<k>] [<lambda>] [<logDir>] [<randStep>] [<est>]
```

Starts algorithm for __iter__ iterations on images in directory <path> with specified file extension <ext>.  
Parameter __k__ ([1, #Images], default = 1) defines how many images shall be compared each iteration (k-best).  
Model parameter __lambda__ ([0,1], default = 1) influences the computation of confidence measures (see [algorithm](http://www.mpi-inf.mpg.de/~kkim/mgc/MainPaper.pdf) for details).   
__logDir__ sets the path for the logfile (default = "log/matchGraph.log").  
Each __randStep__-th iteration, the algorithm uses <k> random image pairs to be compared.  
__est__ chooses estimator (0 = random columns estimator, 1 = global k-best estimator, default = 0).

#### 2. Random execution:
```bash
    Main -r <dim> <k> <iter> [<lambda>]
```    

In this mode, no image comparison is done. The matrix representation gets updated with random similiar/dissimilar results in each iteration for the estimated k-best image-pairs.  
__dim__ defines the size of the simulated matrix representation.  
Other parameters same as above.

## Contributors

Armin Gufler, Julio Rodrigues, Fabian Schwarzkopf
