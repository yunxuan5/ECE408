// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// scan kernel
__global__ void scan(float *input, float *output, int len, int sumFlag) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  int tx = threadIdx.x;
  int i, load;
  if(!sumFlag){ //for first scan 
    i = 2 * blockDim.x * blockIdx.x + tx; 
    load = blockDim.x;
  }else{ //for sum scan
    i = 2 * blockDim.x * (tx + 1) - 1;  //each thread handle 2 element, reaching the end of thread in each segment
    load = 2 * blockDim.x;  // go to the next block
  }

  if(i < len){
    T[tx] = input[i];  // load first block into shared memory
  }else{
    T[tx] = 0.0f;
  }

  if(i + load < len){
    T[tx + blockDim.x] = input[i + load];  // load second block into shared memory
  }
  else{
    T[tx + blockDim.x] = 0.0f;
  }

  // int tx = threadIdx.x;
  // int start = 2 * blockDim.x * blockIdx.x;
  // T[2 * tx] = (start + 2 * tx < len) ? input[start + 2 * tx ] : 0;
  // T[2 * tx + 1] = (start + 2 * tx + 1 < len) ? input[start + 2 * tx + 1] : 0;
  
  // Brent-Kung scan step
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE){
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;

    if(index < (2 * BLOCK_SIZE) && (index - stride) >= 0){
      T[index] += T[index - stride];
    }
    
    stride = stride * 2;
  }

  // post scan step
  stride = BLOCK_SIZE / 2;
  while(stride > 0){
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;

    if((index + stride) < (2 * BLOCK_SIZE)){
      T[index + stride] += T[index];
    }

    stride = stride / 2;
  }

  __syncthreads();
  // store black to global memory
  int str_start = 2 * blockIdx.x * blockDim.x + tx;
  if(str_start < len){
    output[str_start] = T[tx];
  }

  if(str_start + blockDim.x < len){
    output[str_start + blockDim.x] = T[tx + blockDim.x];
  }
}

// add kernel
__global__ void add(float *input, float *output, float *sum, int len) {
  int tx = threadIdx.x;
  int index = 2*blockIdx.x*blockDim.x + tx;

  __shared__ float result;

  if(tx == 0){
    if(blockIdx.x == 0){
      result = 0; //if first thread in first block, add nothing
    }else{
      result = sum[blockIdx.x - 1];
    }
  }

  __syncthreads();

  if(index < len){
    output[index] = input[index] + result;
  }

  if(index + blockDim.x < len){
    output[index+ blockDim.x] = input[index+ blockDim.x] + result;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceScanOutput;
  float *sumArray;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&sumArray, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numElements- 1)/BLOCK_SIZE + 1, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGridtmp(1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceScanOutput, numElements, 0); //first scan
  scan<<<DimGridtmp, DimBlock>>>(deviceScanOutput, sumArray, numElements, 1); //sum scan
  add<<<DimGrid, DimBlock>>>(deviceScanOutput, deviceOutput, sumArray, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
