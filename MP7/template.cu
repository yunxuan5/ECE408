// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void to_char(float* input, unsigned char* output, int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    output[index] = (unsigned char)(255 * input[index]);
  }
}

__global__ void to_float(unsigned char* input, float* output, int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    output[index] = (float)(input[index] / 255.0);
  }
}

__global__ void to_grayscale(unsigned char* input, unsigned char* output, int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    output[index] = (unsigned char)(0.21 * input[3 * index] + 0.71 * input[3 * index + 1] + 0.07 * input[3 * index + 2]);
  }
}

__global__ void compute_histogram(unsigned char* input, unsigned int* output, int length){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ unsigned int histogram[HISTOGRAM_LENGTH];
  
  if(threadIdx.x < HISTOGRAM_LENGTH){
    histogram[threadIdx.x] = 0;
  }
  __syncthreads();

  if(index < length){
    atomicAdd(&(histogram[input[index]]), 1);
  }
  __syncthreads();

  if(threadIdx.x < HISTOGRAM_LENGTH){
    atomicAdd(&(output[threadIdx.x]), histogram[threadIdx.x]);
  }
}

__global__ void histogram_cdf(unsigned int* input, float* result, int size){
  __shared__ float cdf[2 * HISTOGRAM_LENGTH];
  int i = threadIdx.x;
  int load = blockDim.x;

  if(i < HISTOGRAM_LENGTH){
    cdf[i] = input[i];  // load first block into shared memory
  }else{
    cdf[i] = 0.0f;
  }

  if(i + load < HISTOGRAM_LENGTH){
    cdf[i + load] = input[i + load];  // load second block into shared memory
  }
  else{
    cdf[i + load] = 0.0f;
  }

  // Brent-Kung scan step
  int stride = 1;
  while(stride <= HISTOGRAM_LENGTH * 2){
    __syncthreads();
    int index = (i + 1) * stride * 2 - 1;

    if(index < HISTOGRAM_LENGTH * 2){
      cdf[index] += cdf[index - stride];
    }

    stride = stride * 2;
  }

  // post scan step
  stride = HISTOGRAM_LENGTH / 2;
  while(stride > 0){
    __syncthreads();
    int index = (i + 1) * stride * 2 - 1;

    if((index + stride) < HISTOGRAM_LENGTH * 2){
      cdf[index + stride] += cdf[index];
    }

    stride = stride / 2;
  }

  __syncthreads();
  if(i < HISTOGRAM_LENGTH){
    result[i] = ((float) cdf[i] / (size));  
  }
  if(i + blockDim.x < HISTOGRAM_LENGTH){
    result[i + blockDim.x] = ((float) cdf[i + blockDim.x] / (size));  
  }
}

__global__ void equalize_histogram(unsigned char* input, float* cdf, int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    input[index] = (unsigned char) min(max((255 * cdf[input[index]] - cdf[0]) / (1 - cdf[0]), 0.0), 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceImageCharGrayScale;
  unsigned int *deviceImageHistogram;
  float *deviceImageCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **)&deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceImageChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageCharGrayScale, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float));
  
  cudaMemcpy(deviceImageFloat, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  // to char
  dim3 DimGrid1((imageWidth * imageHeight * imageChannels - 1) / 256 + 1, 1, 1);
  dim3 DimBlock1(256, 1, 1);
  to_char<<<DimGrid1, DimBlock1>>>(deviceImageFloat, deviceImageChar, imageWidth * imageHeight * imageChannels);

  cudaDeviceSynchronize();

  // grayscale
  dim3 DimGrid2((imageWidth * imageHeight - 1) / 256 + 1, 1, 1);
  dim3 DimBlock2(256, 1, 1);
  to_grayscale<<<DimGrid2, DimBlock2>>>(deviceImageChar, deviceImageCharGrayScale, imageWidth * imageHeight);

  cudaDeviceSynchronize();

  // compute histogram
  dim3 DimGrid3((imageWidth * imageHeight - 1) / 256 + 1, 1, 1);
  dim3 DimBlock3(256, 1, 1);
  compute_histogram<<<DimGrid3, DimBlock3>>>(deviceImageCharGrayScale, deviceImageHistogram, imageWidth * imageHeight);

  cudaDeviceSynchronize();

  // compute cdf
  dim3 DimGrid4(1, 1, 1);
  dim3 DimBlock4(128, 1, 1);
  histogram_cdf<<<DimGrid4, DimBlock4>>>(deviceImageHistogram, deviceImageCDF, imageWidth * imageHeight);

  cudaDeviceSynchronize();
  
  // equlization
  dim3 DimGrid5((imageWidth * imageHeight * imageChannels - 1) / 256 + 1, 1, 1);
  dim3 DimBlock5(256, 1, 1);
  equalize_histogram<<<DimGrid5, DimBlock5>>>(deviceImageChar, deviceImageCDF, imageWidth * imageHeight * imageChannels);

  cudaDeviceSynchronize();

  // to float
  dim3 DimGrid6((imageWidth * imageHeight * imageChannels - 1) / 512 + 1, 1, 1);
  dim3 DimBlock6(512, 1, 1);
  to_float<<<DimGrid6, DimBlock6>>>(deviceImageChar, deviceImageFloat, imageWidth * imageHeight * imageChannels);

  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageChar);
  cudaFree(deviceImageCharGrayScale);
  cudaFree(deviceImageHistogram);
  cudaFree(deviceImageCDF);

  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
