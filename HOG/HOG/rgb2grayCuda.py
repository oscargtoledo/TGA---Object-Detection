import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import skimage.io, skimage.color
import matplotlib.pyplot as plt
import math
import time
import sys

image = skimage.io.imread("fotofamilia - Copy.bmp")
# image = skimage.color.rgb2gray(image)
image = skimage.img_as_float32(image)

image = np.array(image)
# image = image.astype(np.float32)

print(image.shape)




# d_imageOut = np.empty_like(image)




HOGModule = SourceModule("""
__global__ void doublify(float *in, float *out, int Ncol, int Nrow)
  {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = fil*Ncol + col;
    if (fil < Nrow && col < Ncol)
      out[ind] = in[ind];
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
  __global__ void calculate_gradient(float *in, float *out, int Ncol, int Nrow, int Ncar, bool templateType){
    
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      int fil = blockIdx.y * blockDim.y + threadIdx.y;
      int car = blockIdx.z * blockDim.z + threadIdx.z;
      int ind = Ncar * (fil*Ncol + col);
      if (fil < Nrow && col < Ncol){
        // Si es la primera o ultima fila, o la primera o ultima columna, no calculem gradient perque es el marge
        if (fil == 1 || fil == Nrow-1 || col == 1 || col == Ncol-1){
          out[ind+0] = 1.0f;
          out[ind+1] = 0.0f;
          out[ind+2] = 0.0f;
          return;
        } else if (fil>1 || col >1){
          // Si no, calculem gradient

          // templateType == True -> horitzontal , False -> vertical
          // Per al pixel a la posicio (fil,col)

          float sum = 0;
          //for(int i = -1; i<=1; i++){
              //out[ind+i] = in[ind+i];
              //sum += in[ind+i*Ncar] * i;
          //}
          if(templateType){
            sum += in[Ncar * ((fil)*Ncol + col-1)] * -1;
            sum += in[Ncar * ((fil)*Ncol + col)] * 0;
            sum += in[Ncar * ((fil)*Ncol + col+1)] * 1;

            for(int i = 0; i<3; i++){
              out[ind+i] = sum;
            }
          } else {
            for(int i = 0; i<3; i++){
             out[ind+i] = in[ind+i];
          }
          }
          


          




        }  
      }
  }
""")

esquema = """__global__ void calculate_gradient(float* imageIn, float* imageOut, int Ncol, int Nrow, int Ncar, bool templateType )
    {
      int ts = 3;

      int row = 0; //calcular una row entre [(ts-1)/2.0 , NRow + (ts-1)/2.0 )
      int col = 0; //calcular una col entre [(ts-1)/2.0 , NCol + (ts-1)/2.0 )
      float currentRegion[3][3];  //extreure tall de la imatge, amb les rows:
                                  //r-(ts-1)/2.0 : r+(ts-1)/2.0
                                  //i columnes:
                                  //c-(ts-1)/2.0 : c+(ts-1)/2.0
      float template[3] = {-1,0,1}
      float regionResults[3][3];
      if(templateType) //template vertical
      {
        //interpretar template com una matriu [3][1] i multiplicar contra  current region
        //regionResult[3][3] = currentRegion * template;
        
      } else { //template horitzontal
        //interpretar template com una matriu [1][3] i multiplicar contra  current region
        //regionResult[3][3] = currentRegion * template;
      }
      //float sum = suma de tots els elements de regionResults
      //imageOut[row][col] = sum
    }"""

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


d_imageIn = cuda.mem_alloc(image.nbytes)
d_imageOut = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(d_imageIn,image)
cuda.memcpy_htod(d_imageOut,np.empty_like(image))


## Convertir imatge a grayscale ----------------------------------------------------------------------
func = HOGModule.get_function("rgb2gray")
start = time.time()
func(d_imageIn,d_imageOut,np.int32(Ncol),np.int32(Nrow), np.int32(Ncar),block=dimBlock,grid=gridDim)
end = time.time()
print("Elapsed Time (RGB to Grayscale Conversion): " + str(end-start))


h_image = np.empty_like(image)
cuda.memcpy_dtoh(h_image,d_imageOut)
skimage.io.imshow(h_image)
plt.show()

## Resaltar arestes imatge ----------------------------------------------------------------------

## La imatge final ha de tenir un pixel extra per cada costat, per tant el tamany incrementa: 

nBlocksCol = int((Ncol+2 + nThreads -1)/nThreads)
nBlocksRow = int((Nrow+2 + nThreads -1)/nThreads)
nBlocksCar = int((Ncar + nThreads -1)/nThreads)


gridDim = (nBlocksCol,nBlocksRow,nBlocksCar)

preparedImage = np.zeros((Nrow+2,Ncol+2,Ncar), dtype=np.float32)
y_offset = 1
x_offset = 1
preparedImage[y_offset:h_image.shape[0]+y_offset,x_offset:h_image.shape[1]+x_offset] = image

d_imageIn = cuda.mem_alloc(preparedImage.nbytes)
d_imageOut = cuda.mem_alloc(preparedImage.nbytes)
cuda.memcpy_htod(d_imageIn,preparedImage)
cuda.memcpy_htod(d_imageOut,np.empty_like(preparedImage))

HOGFunc = HOGModule.get_function("calculate_gradient")
HOGFunc(d_imageIn,d_imageOut,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar), np.int32(1) ,block=dimBlock,grid=gridDim)

h_image = np.empty_like(preparedImage)
cuda.memcpy_dtoh(h_image,d_imageOut)
skimage.io.imshow(h_image)
plt.show()


# d_imageIn = cuda.mem_alloc(image.nbytes)
# # d_imageOut = cuda.mem_alloc(image.nbytes)
# d_imageOut = cuda.mem_alloc(image.nbytes + 2*image[0].nbytes + 2*image[1].nbytes)
# # d_imageOut = cuda.mem_alloc(image.nbytes + 2*image.shape[0]*Ncar*sys.getsizeof(float()) + 2*image.shape[1]*Ncar*sys.getsizeof(float()))
# cuda.memcpy_htod(d_imageIn,image)
# # cuda.memcpy_htod(d_imageOut,np.empty_like(image))
# cuda.memcpy_htod(d_imageOut,np.zeros((Nrow+2,Ncol+2,Ncar), dtype=np.float32))

# imgCopy = HOGModule.get_function("imageCopy")
# imgCopy(d_imageIn,d_imageOut,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar),block=dimBlock,grid=gridDim)

# h_image = np.zeros((Nrow,Ncol,Ncar), dtype=np.float32)
# cuda.memcpy_dtoh(h_image,d_imageOut)
# skimage.io.imshow(h_image)
# plt.show()








# d_finalImage = cuda.mem_alloc(image.nbytes + 2*image.shape[0]*Ncar*sys.getsizeof(float()) + 2*image.shape[1]*Ncar*sys.getsizeof(float()))
# cuda.memcpy_htod(d_imageIn,h_image)
# print(np.zeros((Nrow+2,Ncol+2,Ncar)))
# cuda.memcpy_htod(d_finalImage,np.zeros((Nrow+2,Ncol+2,Ncar), dtype=np.float32))
# imgCopy(d_imageIn,d_finalImage,np.int32(Ncol),np.int32(Nrow), np.int32(Ncar),block=dimBlock,grid=gridDim)

# h_result = np.zeros((Nrow+2,Ncol+2,Ncar), dtype=np.float32)
# cuda.memcpy_dtoh(h_result, d_imageOut)
# skimage.io.imshow(h_result)
# plt.show()





# image = skimage.io.imread("fotofamilia - Copy.bmp")
# image = skimage.color.rgb2gray(image)
# skimage.io.imshow(image)
# plt.show()


