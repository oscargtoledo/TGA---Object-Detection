import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import skimage.io, skimage.color
import matplotlib.pyplot as plt
import math


image = skimage.io.imread("fotofamilia - Copy.bmp")
# image = skimage.color.rgb2gray(image)
image = skimage.img_as_float32(image)

image = np.array(image)
# image = image.astype(np.float32)

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


  __global__ void imageCopy(float *in, float *out, int Ncol, int Nrow, int Ncar){
    
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      int fil = blockIdx.y * blockDim.y + threadIdx.y;
      int car = blockIdx.z * blockDim.z + threadIdx.z;
      //int ind = car * Nrow * Ncol + fil*Ncol + col;
      //if (fil < Nrow && col < Ncol && car < Ncar && car == 0){
      int ind = Ncar * (fil*Ncol + col);
      if (fil < Nrow && col < Ncol){
        for(int i = 0; i<3; i++){
          
          out[ind+i] = in[ind+i];
        }
      }
    
    
  }

  __global__ void rgb2gray(float *in, float *out, int Ncol, int Nrow, int Ncar){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = Ncar *(fil*Ncol + col);
    if (fil < Nrow && col < Ncol){

      float clinear = 0.2126 * in[ind] + 0.7152 * in[ind+1] + 0.0722 * in[ind+2];
      float csrgb = 0.0;
      if(clinear<=0.0031308){
        csrgb = 12.92*clinear;
      } else {
        csrgb = (1.055*pow((double)clinear,1/2.4))-0.055;
      }

      out[ind] = csrgb;
      out[ind+1] = csrgb;
      out[ind+2] = csrgb;

            
        
    }
  }

  """)

N = image.shape[0]

Nrow, Ncol, Ncar = image.shape
# THREADS = 16


nThreads = 8
# Ncol = 720
# Nrow = 122
# Ncar = 1
nBlocksCol = int((Ncol + nThreads -1)/nThreads)
nBlocksRow = int((Nrow + nThreads -1)/nThreads)
nBlocksCar = int((Ncar + nThreads -1)/nThreads)
# nBlocks = math.floor(N+1/nThreads)

gridDim = (nBlocksCol,nBlocksRow,nBlocksCar)
# gridDim = (1,1,1)
dimBlock = (nThreads, nThreads, nThreads)


func = module.get_function("rgb2gray")
func(d_imageIn,d_imageOut,np.int32(Ncol),np.int32(Nrow), np.int32(Ncar),block=dimBlock,grid=gridDim)
# func.prepare("P")
# func.prepared_call((1,1),(4,4,1),[d_imageIn,d_imageOut])

h_image = np.empty_like(image)
cuda.memcpy_dtoh(h_image,d_imageOut)
print(h_image)
print("--------------------------------")
print(image)
# print(d_image)

# skimage.io.imsave("e.png",h_image)
skimage.io.imshow(h_image)
plt.show()

image = skimage.io.imread("fotofamilia - Copy.bmp")
image = skimage.color.rgb2gray(image)
skimage.io.imshow(image)
plt.show()


