import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import skimage.io, skimage.color
import matplotlib.pyplot as plt
import math
import time
import sys
import argparse

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Parameters for HOG generation")
  parser.add_argument('-file',type=str,default="campus.jpg", help="File image to be processed")
  parser.add_argument('-showImages',type=bool,default=False, help="Whether step images will be shown")

  args = parser.parse_args()

  file = args.file
  showImages = args.showImages



  image = skimage.io.imread(file)
  image = skimage.img_as_float32(image)
  image = np.array(image)


  HOGModule = SourceModule(open("HOGSource.cu",'r').read())


  N = image.shape[0]
  Nrow, Ncol, Ncar = image.shape


  nThreads = 16
  nBlocksCol = int((Ncol + nThreads -1)/nThreads)
  nBlocksRow = int((Nrow + nThreads -1)/nThreads)
  nBlocksCar = int((Ncar + nThreads -1)/nThreads)


  gridDim = (nBlocksCol,nBlocksRow,1)
  dimBlock = (nThreads, nThreads, 1)







  ## Convertir imatge a grayscale ----------------------------------------------------------------------
  d_imageIn = cuda.mem_alloc(image.nbytes)
  z = np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)

  d_imageOut = cuda.mem_alloc(z.nbytes)
  cuda.memcpy_htod(d_imageIn,image)
  cuda.memcpy_htod(d_imageOut,np.empty_like(z))


  func = HOGModule.get_function("rgb2gray")
  start = time.time()
  func(d_imageIn,d_imageOut,np.int32(Ncol),np.int32(Nrow), np.int32(Ncar),block=dimBlock,grid=gridDim)
  end = time.time()
  print("Elapsed Time (RGB to Grayscale Conversion): " + str(end-start))


  h_image = np.empty_like(z)
  cuda.memcpy_dtoh(h_image,d_imageOut)
  if showImages:
    skimage.io.imshow(h_image, cmap=plt.cm.gray)
    plt.show()
  # quit()


  ## Resaltar arestes imatge ----------------------------------------------------------------------
  ## La imatge final ha de tenir un pixel extra per cada costat, per tant el tamany incrementa: 

  nBlocksCol = int((Ncol+2 + nThreads -1)/nThreads)
  nBlocksRow = int((Nrow+2 + nThreads -1)/nThreads)
  nBlocksCar = int((Ncar + nThreads -1)/nThreads)


  gridDim = (nBlocksCol,nBlocksRow,nBlocksCar)

  preparedImage = np.zeros((Nrow+2,Ncol+2), dtype=np.float32)
  y_offset = 1
  x_offset = 1
  preparedImage[y_offset:h_image.shape[0]+y_offset,x_offset:h_image.shape[1]+x_offset] = h_image


  d_grayscaleIn = cuda.mem_alloc(preparedImage.nbytes)
  d_vGradient = cuda.mem_alloc(preparedImage.nbytes)
  d_hGradient = cuda.mem_alloc(preparedImage.nbytes)

  cuda.memcpy_htod(d_grayscaleIn,preparedImage)
  cuda.memcpy_htod(d_vGradient,np.empty_like(preparedImage))
  cuda.memcpy_htod(d_hGradient,np.empty_like(preparedImage))

  HOGFunc = HOGModule.get_function("calculate_gradient")
  HOGFunc(d_grayscaleIn,d_vGradient,np.int32(preparedImage.shape[1]),np.int32(preparedImage.shape[0]), np.int32(Ncar), np.int32(0) ,block=dimBlock,grid=gridDim)
  HOGFunc(d_grayscaleIn,d_hGradient,np.int32(preparedImage.shape[1]),np.int32(preparedImage.shape[0]), np.int32(Ncar), np.int32(1) ,block=dimBlock,grid=gridDim)

  h_vGradient = np.empty_like(preparedImage)
  h_hGradient = np.empty_like(preparedImage)
  cuda.memcpy_dtoh(h_vGradient,d_vGradient)
  cuda.memcpy_dtoh(h_hGradient,d_hGradient)

  if showImages:
    skimage.io.imshow(h_vGradient, cmap=plt.cm.gray)
    plt.show()
    skimage.io.imshow(h_hGradient, cmap=plt.cm.gray)
    plt.show()

  ## Generar imatges amb magnitud i direcció de gradient ----------------------------------------

  d_gradMag = cuda.mem_alloc(preparedImage.nbytes)
  d_gradDir = cuda.mem_alloc(preparedImage.nbytes)
  cuda.memcpy_htod(d_gradMag,np.empty_like(preparedImage))
  cuda.memcpy_htod(d_gradDir,np.empty_like(preparedImage))

  HOGgradDir = HOGModule.get_function("gradient_direction")
  HOGmagDir = HOGModule.get_function("gradient_magnitude")

  HOGgradDir(d_hGradient,d_vGradient,d_gradDir,np.int32(preparedImage.shape[1]),np.int32(preparedImage.shape[0]), np.int32(Ncar), block=dimBlock,grid=gridDim)
  HOGmagDir(d_hGradient,d_vGradient,d_gradMag,np.int32(preparedImage.shape[1]),np.int32(preparedImage.shape[0]), np.int32(Ncar), block=dimBlock,grid=gridDim)

  h_gradMag = np.empty_like(preparedImage)
  h_gradDir = np.empty_like(preparedImage)
  cuda.memcpy_dtoh(h_gradMag,d_gradMag)
  cuda.memcpy_dtoh(h_gradDir,d_gradDir)

  if showImages:
    skimage.io.imshow(h_gradDir)
    plt.show()
    skimage.io.imshow(h_gradMag)
    plt.show()

  #Normalized magnitude gradient
  h_gradMag = (h_gradMag - np.min(h_gradMag))/np.ptp(h_gradMag).astype(float)

  ## Generar histograma -------------------------------------------------


  histogram = np.zeros((gridDim[0]*gridDim[1],9))
  d_histogram = cuda.mem_alloc(histogram.nbytes)



  HOGHisto = HOGModule.get_function("histogram")
  HOGHisto(d_gradDir,d_gradMag,d_histogram,np.int32(preparedImage.shape[1]),np.int32(preparedImage.shape[0]), np.int32(Ncar), block=dimBlock,grid=gridDim)


  h_histogram = np.empty_like(histogram)
  cuda.memcpy_dtoh(h_histogram,d_histogram)



  h_histogram = (h_histogram - np.min(h_histogram))/np.ptp(h_histogram).astype(float)

  for h in h_histogram:

    plt.bar(x=np.arange(9),height=h)
    plt.ylim((0,1))

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



