#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 3
#define MASK_SIZE 3
#define MASK_RADIUS (MASK_SIZE/2)
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_SIZE][MASK_SIZE][MASK_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  float result = 0.0f;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  __shared__ float N_ds[TILE_WIDTH + MASK_SIZE - 1][TILE_WIDTH + MASK_SIZE - 1][TILE_WIDTH + MASK_SIZE - 1];
  
  int out_x = bx * TILE_WIDTH + tx;
  int out_y = by * TILE_WIDTH + ty;
  int out_z = bz * TILE_WIDTH + tz;

  int start_x = out_x - MASK_RADIUS;
  int start_y = out_y - MASK_RADIUS;
  int start_z = out_z - MASK_RADIUS;

  if((start_x >= 0 && start_x < x_size) && (start_y >= 0 && start_y < y_size) && (start_z >= 0 && start_z < z_size)){
    N_ds[tz][ty][tx] = input[start_x + start_y * x_size + start_z * x_size * y_size];
  }
  else{
    N_ds[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  if(tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
    for(int c = 0; c < MASK_SIZE; c++){
      for(int b = 0; b < MASK_SIZE; b++){
        for(int a = 0; a < MASK_SIZE; a++){
          result += deviceKernel[c][b][a] * N_ds[tz + c][ty + b][tx + a];
        }
      }
    }
    if(out_x < x_size && out_y < y_size && out_z < z_size){
      output[out_x + out_y * x_size + out_z * x_size *y_size] = result;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void**) &deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid((x_size - 1)/ TILE_WIDTH + 1, (y_size - 1)/ TILE_WIDTH + 1, (z_size - 1)/ TILE_WIDTH + 1);
  dim3 DimBlock(TILE_WIDTH + MASK_SIZE - 1, TILE_WIDTH + MASK_SIZE - 1, TILE_WIDTH + MASK_SIZE - 1);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
