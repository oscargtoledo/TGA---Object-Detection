  __global__ void rgb2gray(float *in, float *out, int Ncol, int Nrow, int Ncar){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = Ncar *(fil*Ncol + col);
    int ind2 = (fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      float clinear = 0.2126 * in[ind] + 0.7152 * in[ind+1] + 0.0722 * in[ind+2];
      float csrgb = 0.0;
      if(clinear<=0.0031308){
        csrgb = 12.92*clinear;
      } else {
        csrgb = (1.055*pow((double)clinear,1/2.4))-0.055;
      }

      out[ind2] = csrgb;
    }

  }
  __global__ void calculate_gradient(float *in, float *out, int Ncol, int Nrow, int Ncar, bool templateType){
    
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      int fil = blockIdx.y * blockDim.y + threadIdx.y;
      int car = blockIdx.z * blockDim.z + threadIdx.z;
      int ind = (fil*Ncol + col);
      if (fil < Nrow && col < Ncol){

        // Si es la primera o ultima fila, o la primera o ultima columna, no calculem gradient perque es el marge
        if (fil == 1 || fil == Nrow-1 || col == 1 || col == Ncol-1){
          return;
        } else { //if (fil>1 && fil<Nrow-1 || col >1){
          // Si no, calculem gradient

          // templateType == True -> horitzontal , False -> vertical

          

          if(templateType && col > 1 && col < Ncol-1){
            float sum = 0;
            sum += in[((fil)*Ncol + col-1)] * -1;
            sum += in[((fil)*Ncol + col)] * 0;
            sum += in[((fil)*Ncol + col+1)] * 1;

            
            out[ind] = sum;
            
          } else if (!templateType && fil > 1 && fil < Nrow-1){
            float sum = 0;
            sum += in[((fil-1)*Ncol  + col) ] * -1;
            sum += in[((fil)*Ncol    + col)   ] * 0;
            sum += in[((fil+1)*Ncol  + col) ] * 1;

            
            out[ind] = sum;
            

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
    int ind = (fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        float gradDir = 0.0f;
        gradDir = atan(vGrad[ind]/(hGrad[ind]+0.00000001));
        float pi = 3.14159265359f;
        gradDir = gradDir * (180.0f / pi);
        gradDir = fmod(gradDir,180.0f);
        
          out[ind] = gradDir;
        
      }
      
    }
  }

  __global__ void gradient_magnitude(float* hGrad, float* vGrad,float* out, int Ncol, int Nrow, int Ncar)
  {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = (fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        float hGradSq = pow(hGrad[ind],2);
        float vGradSq = pow(vGrad[ind],2);
        float sumSquares = hGradSq + vGradSq;
        float gradMag = sqrt(sumSquares);
        out[ind] = gradMag;
        
      }
    }
    
  }

  __global__ void histogram(float* dir, float* mag, float* hist,int Ncol, int Nrow, int Ncar)
  {
    __shared__ float privateHistogram[8];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int fil = blockIdx.y * blockDim.y + threadIdx.y;

    int histoOffset = blockIdx.y*9+blockIdx.x;
    int ind = (fil*Ncol + col);
    if (fil < Nrow && col < Ncol){
      if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        
        int workThread = threadIdx.x + blockDim.x * threadIdx.y;
        float fMag = mag[ind];
        float fDir = abs(dir[ind]);
        
        // Si te part decimal, esta entre dos caselles d'histograma
        if( abs(fDir/20-int(fDir/20))>0 )
        {

          atomicAdd(&hist[histoOffset+workThread],int(floor(fDir/20)));
          atomicAdd(&hist[histoOffset+workThread],int(ceil(fDir/20)));
        } else {
          atomicAdd(&hist[workThread],int(fDir/20));
        }
        return;
        

        // if(workThread <= 8)
        // {
        //   atomicAdd(&hist[histoOffset+workThread],privateHistogram[workThread]);
        // }


      }
    }








  // __global__ void histogram(float* dir, float* mag, float* hist,int Ncol, int Nrow, int Ncar)
  // {
  //   __shared__ float privateHistogram[8];
  //   int col = blockIdx.x * blockDim.x + threadIdx.x;
  //   int fil = blockIdx.y * blockDim.y + threadIdx.y;
  //   int ind = (fil*Ncol + col);
  //   if (fil < Nrow && col < Ncol){
  //     if (fil != 1 && fil != Nrow-1 && col != 1 && col != Ncol-1){
        
  //       int workThread = threadIdx.x + blockDim.x * threadIdx.y;
  //       //printf("%f \\n", ind);
  //       //printf("%f %f -> %f\\n",threadIdx.x, threadIdx.y, workThread);
  //       //if(workThread <= 8)
  //       //  privateHistogram[workThread] = 0;
  //       //__syncthreads();

  //       float fMag = mag[ind];
  //       float fDir = abs(dir[ind]);
        
        
  //       // Si te part decimal, esta entre dos caselles d'histograma
  //       if( abs(fDir/20-int(fDir/20))>0 )
  //       {
          
  //         //atomicAdd(&privateHistogram[int(floor(fDir/20))],fMag);
  //         //atomicAdd(&privateHistogram[int(ceil(fDir/20))],fMag);

  //         atomicAdd(&hist[workThread],int(floor(fDir/20)));
  //         atomicAdd(&hist[workThread],int(ceil(fDir/20)));
  //       } else {
  //         //atomicAdd(&privateHistogram[int(fDir/20)],fMag);

  //         atomicAdd(&hist[workThread],int(fDir/20));
  //       }
  //       return;
  //       __syncthreads();

  //       if(workThread <= 8)
  //       {
  //         atomicAdd(&hist[workThread],privateHistogram[workThread]);
  //       }


  //     }
  //   }
  }
