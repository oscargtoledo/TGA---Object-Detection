import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import skimage.io, skimage.color
import matplotlib.pyplot as plt
import math


image = skimage.io.imread("fotofamiliaSquare.jpg")
image = skimage.color.rgb2gray(image)

image = np.array(image)
image = image.astype(np.float32)

print(image.shape)



d_imageIn = cuda.mem_alloc(image.nbytes)
d_imageOut = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(d_imageIn,image)
cuda.memcpy_htod(d_imageOut,np.empty_like(image))
# d_imageOut = np.empty_like(image)



module = SourceModule("""
  __global__ void doublify(float *in, float *out, int Ncol, int Nrow)
  {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = fil*Ncol + col;
    if (fil < Nrow && col < Ncol)
      out[ind] = in[ind];
  }
  """)

N = image.shape[0]

Ncol = image.shape[0]
Nrow = image.shape[1]
# THREADS = 16


nThreads = 32
nBlocks = int((image.shape[0] + nThreads -1)/nThreads)
# nBlocks = math.floor(N+1/nThreads)

gridDim = (nBlocks,nBlocks,1)
dimBlock = (nThreads, nThreads, 1)

func = module.get_function("doublify")
func(d_imageIn,d_imageOut,np.int32(Ncol),np.int32(Nrow),block=dimBlock,grid=gridDim)
# func.prepare("P")
# func.prepared_call((1,1),(4,4,1),[d_imageIn,d_imageOut])

h_image = np.empty_like(image)
cuda.memcpy_dtoh(h_image,d_imageOut)
print(h_image)
# print(d_image)

# skimage.io.imsave("e.png",h_image)
skimage.io.imshow(h_image,cmap=plt.cm.gray)
plt.show()

