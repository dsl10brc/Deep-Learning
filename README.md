# Deep-Learning software installation for MacOS -> keras, theano, tensorflow, CUDA 8.0

##Installation of CUDA

https://developer.nvidia.com/cuda-downloads   -  download and install

##Installation of cudnn

https://developer.nvidia.com/cudnn

Once you have it downloaded locally, you can unzip — All this should be in home folder (i.e include and lib folders should be outside cuda - extracted folder)

Then use these commands 
```
sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib
sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/

```

##add to .bashrc file
```
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"

```
##create and add to .theanorc file
```
[global]
device = gpu
floatX = float32
optimizer_including = cudnn
[cuda]
root = /usr/local/cuda
[nvcc]
fastmath = True

[blas]
ldflags = -llapack -lblas

[cmodule]
mac_framework_link=True

#[lib]
#cnmem=.10

```
##for tensor flow installation:

https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation

possible error you might get: 
```
———————————————x—————————————————x———————————————————x——————————————————————————
if you get error like this:

I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.7.5.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.5.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.7.5.dylib locally
I tensorflow/stream_executor/dso_loader.cc:102] Couldn't open CUDA library libcuda.1.dylib. LD_LIBRARY_PATH: :/usr/local/cuda/lib
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:160] hostname: sfocfjgn32.ads.autodesk.com
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:185] libcuda reported version is: Not found: was unable to find libcuda.so DSO loaded into this program
Segmentation fault: 11

then do:

$ cd /usr/local/cuda/lib
$ ln -s libcuda.dylib libcuda.1.dylib

———————————————x—————————————————x———————————————————x——————————————————————————
```

##install keras
`sudo pip install keras`

##to change from theano to tensorflow
 `vi ~/.keras/keras.json`

##for keras to use theano
```
{
"image_dim_ordering": "th",
"epsilon": 1e-07,
"floatx": "float32",
"backend": "theano"
}
```

##for keras to use tensorflow
```
{
"image_dim_ordering": "tf”,
"epsilon": 1e-07,
"floatx": "float32",
"backend": "tensorflow"
}
```

##run this code to check if it uses gpu and also if it uses theano or tensor flow
`python -c "import keras; print(keras.__version__)"`



##run this code to check if it uses gpu
```
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
```
