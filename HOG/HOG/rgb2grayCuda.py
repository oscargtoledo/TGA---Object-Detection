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
        } else { //if (fil>1 && fil<Nrow-1 || col >1){
          // Si no, calculem gradient

          // templateType == True -> horitzontal , False -> vertical

          

          if(templateType && col > 1 && col < Ncol-1){
            float sum = 0;
            sum += in[Ncar * ((fil)*Ncol + col-1)] * -1;
            sum += in[Ncar * ((fil)*Ncol + col)] * 0;
            sum += in[Ncar * ((fil)*Ncol + col+1)] * 1;

            for(int i = 0; i<3; i++){
              out[ind+i] = sum;
            }
          } else if (!templateType && fil > 1 && fil < Nrow-1){
            float sum = 0;
            sum += in[Ncar * ((fil-1)*Ncol  + col) ] * -1;
            sum += in[Ncar * ((fil)*Ncol    + col)   ] * 0;
            sum += in[Ncar * ((fil+1)*Ncol  + col) ] * 1;

            for(int i = 0; i<3; i++){
              out[ind+i] = sum;
            }

            //for(int i = 0; i<3; i++){
            //  out[ind+i] = in[ind+i];
            //}
          }
        }  
      }
  }


  __global__ void gradient_direction(float* hGrad, float* vGrad,float* out, int Ncol, int Nrow, int Ncar)
  {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = Ncar *(fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        float gradDir = 0.0f;
        gradDir = atan(vGrad[ind]/(hGrad[ind]+0.00000001));
        float pi = 3.14159265359f;
        gradDir = gradDir * (180.0f / pi);
        gradDir = fmod(gradDir,180.0f);
        for(int i = 0; i<3; i++){
          out[ind+i] = gradDir;
        }
      }
      
    }
  }

  __global__ void gradient_magnitude(float* hGrad, float* vGrad,float* out, int Ncol, int Nrow, int Ncar)
  {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = Ncar *(fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        float gradMag = 0.0f;
        float hGradSq = pow(hGrad[ind],2);
        float vGradSq = pow(vGrad[ind],2);
        float sumSquares = hGradSq + vGradSq;
        gradMag = sqrt(sumSquares);
        for(int i = 0; i<3; i++){
          out[ind+i] = gradMag;
        }
      }
    }
    
  }

  __global__ void histogram(float* dir, float* mag, float* hist,int Ncol, int Nrow, int Ncar)
  {
    __shared__ float privateHistogram[8];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = Ncar *(fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        
        int workThread = threadIdx.x + blockDim.x * threadIdx.y;
        //printf("%f \\n", ind);
        //printf("%f %f -> %f\\n",threadIdx.x, threadIdx.y, workThread);
        //if(workThread <= 8)
        //  privateHistogram[workThread] = 0;
        //__syncthreads();

        float fMag = mag[ind];
        float fDir = abs(dir[ind]);
        
        
        // Si te part decimal, esta entre dos caselles d'histograma
        if( abs(fDir/20-int(fDir/20))>0 )
        {
          
          //atomicAdd(&privateHistogram[int(floor(fDir/20))],fMag);
          //atomicAdd(&privateHistogram[int(ceil(fDir/20))],fMag);

          atomicAdd(&hist[workThread],int(floor(fDir/20)));
          atomicAdd(&hist[workThread],int(ceil(fDir/20)));
        } else {
          //atomicAdd(&privateHistogram[int(fDir/20)],fMag);

          atomicAdd(&hist[workThread],int(fDir/20));
        }
        return;
        __syncthreads();

        
       
        //printf("%f %f %f %f %f %f %f %f %f \\n", privateHistogram[0],privateHistogram[1],privateHistogram[2],privateHistogram[3],privateHistogram[4],privateHistogram[5],privateHistogram[6],privateHistogram[7],privateHistogram[8]);

        if(workThread <= 8)
        {
          atomicAdd(&hist[workThread],privateHistogram[workThread]);
          //printf("Adding %f to bracket %f\\n",privateHistogram[workThread],workThread);
        }


      }
    }
  }
""")


N = image.shape[0]

Nrow, Ncol, Ncar = image.shape
# THREADS = 16


nThreads = 32
# Ncol = 720
# Nrow = 122
# Ncar = 1
nBlocksCol = int((Ncol + nThreads -1)/nThreads)
nBlocksRow = int((Nrow + nThreads -1)/nThreads)
nBlocksCar = int((Ncar + nThreads -1)/nThreads)
# nBlocks = math.floor(N+1/nThreads)

gridDim = (nBlocksCol,nBlocksRow,nBlocksCar)
# gridDim = (1,1,1)
dimBlock = (nThreads, nThreads, 1)



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
# skimage.io.imshow(h_image, cmap=plt.cm.gray)
# plt.show()


## Resaltar arestes imatge ----------------------------------------------------------------------

## La imatge final ha de tenir un pixel extra per cada costat, per tant el tamany incrementa: 

nBlocksCol = int((Ncol+2 + nThreads -1)/nThreads)
nBlocksRow = int((Nrow+2 + nThreads -1)/nThreads)
nBlocksCar = int((Ncar + nThreads -1)/nThreads)


gridDim = (nBlocksCol,nBlocksRow,nBlocksCar)

preparedImage = np.zeros((Nrow+2,Ncol+2,Ncar), dtype=np.float32)
y_offset = 1
x_offset = 1
preparedImage[y_offset:h_image.shape[0]+y_offset,x_offset:h_image.shape[1]+x_offset] = h_image

d_imageIn = cuda.mem_alloc(preparedImage.nbytes)
d_vGradient = cuda.mem_alloc(preparedImage.nbytes)
d_hGradient = cuda.mem_alloc(preparedImage.nbytes)

cuda.memcpy_htod(d_imageIn,preparedImage)
cuda.memcpy_htod(d_vGradient,np.empty_like(preparedImage))
cuda.memcpy_htod(d_hGradient,np.empty_like(preparedImage))

HOGFunc = HOGModule.get_function("calculate_gradient")
HOGFunc(d_imageIn,d_vGradient,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar), np.int32(0) ,block=dimBlock,grid=gridDim)
HOGFunc(d_imageIn,d_hGradient,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar), np.int32(1) ,block=dimBlock,grid=gridDim)

h_vGradient = np.empty_like(preparedImage)
h_hGradient = np.empty_like(preparedImage)
cuda.memcpy_dtoh(h_vGradient,d_vGradient)
cuda.memcpy_dtoh(h_hGradient,d_hGradient)
# skimage.io.imshow(h_vGradient, cmap=plt.cm.gray)
# plt.show()
# skimage.io.imshow(h_hGradient, cmap=plt.cm.gray)
# plt.show()
## Generar imatges amb magnitud i direcció de gradient ----------------------------------------

d_gradMag = cuda.mem_alloc(preparedImage.nbytes)
d_gradDir = cuda.mem_alloc(preparedImage.nbytes)
cuda.memcpy_htod(d_gradMag,np.empty_like(preparedImage))
cuda.memcpy_htod(d_gradDir,np.empty_like(preparedImage))

HOGgradDir = HOGModule.get_function("gradient_direction")
HOGmagDir = HOGModule.get_function("gradient_magnitude")

HOGgradDir(d_hGradient,d_vGradient,d_gradDir,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar), block=dimBlock,grid=gridDim)
HOGmagDir(d_hGradient,d_vGradient,d_gradMag,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar), block=dimBlock,grid=gridDim)

h_gradMag = np.empty_like(preparedImage)
h_gradDir = np.empty_like(preparedImage)
cuda.memcpy_dtoh(h_gradMag,d_gradMag)
cuda.memcpy_dtoh(h_gradDir,d_gradDir)


# skimage.io.imshow(h_gradDir)
# plt.show()
# skimage.io.imshow(h_gradMag, cmap=plt.cm.gray)
# plt.show()

## Generar histograma -------------------------------------------------
histogram = np.zeros(9)
d_histogram = cuda.mem_alloc(histogram.nbytes)

HOGHisto = HOGModule.get_function("histogram")

HOGHisto(d_gradDir,d_gradMag,d_histogram,np.int32(Ncol+2),np.int32(Nrow+2), np.int32(Ncar), block=dimBlock,grid=gridDim)


h_histogram = np.empty_like(histogram)
cuda.memcpy_dtoh(h_histogram,d_histogram)
print(h_histogram)
plt.bar(x=np.arange(9),height=h_histogram)

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



