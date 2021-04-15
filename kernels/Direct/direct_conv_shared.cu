#include "direct_conv.h"
using namespace std;

__global__ void pad_input(float* f_in, float* f_out, int H, int W, int D, int pad)
{
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int dep = blockIdx.z*blockDim.z+threadIdx.z;

    int new_H = H+2*pad; 
    int new_W = W+2*pad; 
 
    int i = dep * new_H * new_W + col * new_W + row;
    int j = dep * H * W + (col - pad) *W+ (row - pad);

    if(col < new_H && row < new_W && dep < D)
    {
        if((col < pad || col > H+pad-1) || (row < pad || row > W+pad-1)) f_out[i] = 0;
        else f_out[i] = f_in[j];
    }
}

/*parallelization code */
__global__ 
void direct_convolution(int input_channels, int input_height, int input_width, int out_channels, int kernel_height,int kernel_width, 
                        int padding, int stride, int H_out, int W_out, int W_grid, int tile_w, float* X, float* W_filter, float* Y) {
  int n , m , h0 , w0 , h_base, w_base, h , w;
  int X_tile_width = (stride*tile_w) + kernel_width - stride;
  int Y_tile_width = (stride*tile_w) + kernel_height - stride;

  extern __shared__ float shmem[];
  float *X_shared = &shmem[0];
  float *W_shared = &shmem[X_tile_width * Y_tile_width];

  n = blockIdx.x;
  m = blockIdx.y;
  h0 = threadIdx.y;
  w0 = threadIdx.x;
  h_base = (blockIdx.z / W_grid)*tile_w;
  w_base = (blockIdx.z % W_grid)*tile_w;
  h = h_base + h0;
  w = w_base + w0;

  input_height = input_height+padding;
  input_width = input_width+padding;

  if(h<H_out && w<W_out) {
    float temp=0;
    int c, i , j , p , q;
    for(c = 0; c < input_channels; c++) {
        
      for(i = 0; i + h0 < kernel_height; i += tile_w) {
          for(j = 0; j + w0 < kernel_width; j += tile_w) {
              W_shared[(i+h0)*kernel_width + (j+w0)] = W_filter[m*(input_channels*kernel_height*kernel_width) + c*(kernel_height*kernel_width) + (i+h0)*(kernel_height) + (j+w0)];
          }
      }
      __syncthreads();

      for(i = h; i < h_base + Y_tile_width; i += tile_w){
          for(j = w; j < w_base + X_tile_width; j += tile_w){
              X_shared[(i-h_base)*X_tile_width + (j-w_base)] = X[n*(input_channels*input_height*input_width) + c*(input_height*input_width) + i*(input_width) + j];
          }
      }
      __syncthreads();

      for(p = 0; p < kernel_height; p++) {
        for(q = 0; q < kernel_width; q++) {
          temp = temp + X_shared[(h0*stride+p)*X_tile_width + (w0*stride+q)] * W_shared[p*(kernel_height) + q];
        }
      }
      __syncthreads();
    }
    Y[n*(out_channels*H_out*W_out) + m*(H_out*W_out) + h*(W_out) + w] = temp;
  }
}

/*forward pass function declared in direc_conv.hpp library*/
float* Direct::passforward(int out_channels, int input_channels, int kernel_height, int kernel_width, int padding, int stride, 
                          float* weights,int batchsize_of_data, int input_height, int input_width, float* input, float &conv_time, float& overhead_time) {
  if(kernel_height > input_height || kernel_width > input_width){
    cout << "kernel size is too big " << endl;
    exit(EXIT_FAILURE);
  }
  
  conv_time = 0;
  overhead_time = 0;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaError_t err = cudaSuccess;
  
  /* The rest of the code assumes that padding = x means x/2 on either ends hence the modification */
  padding = 2*padding;

  /* Padding */
  int new_input_height = input_height + padding;
  int new_input_width = input_width + padding;

  /* size of matrix with padding*/ 
  int size_input_matrix = batchsize_of_data * input_channels * new_input_height * new_input_width * sizeof(float);   // size of input matrix after padding

  /* size of kernel matrix */ 
  int size_kernel_matrix = out_channels * input_channels * kernel_height * kernel_width * sizeof(float);   // size of the kernal

  /* calculating size of output matrix*/
  int H_out = (new_input_height - kernel_height + stride)/stride;
  int W_out = (new_input_width - kernel_width + stride)/stride;
  int size_output_matrix = batchsize_of_data * out_channels * H_out * W_out * sizeof(float);
  
  /* allocating memory for input  matrix with padding */
  float *h_X = (float*)malloc(size_input_matrix); 
  
  /* allocating memory for output matrix */
  float *h_Y = (float*)malloc(size_output_matrix);
 
  /* memory allocation check*/
  if (h_X == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  float *pad_input_in = NULL; 
  cudaMalloc((void **)&pad_input_in, input_height * input_width * input_channels * sizeof(float));
  
  float *pad_input_out = NULL; 
  cudaMalloc((void **)&pad_input_out, new_input_height * new_input_width * input_channels * sizeof(float));

  dim3 threads1(1, 1, 1);
  dim3 grid1(new_input_height, new_input_width, input_channels);

  for(int i = 0; i < batchsize_of_data; i++)
  {
    cudaMemcpy(pad_input_in, &input[i * input_channels * input_height * input_width],
              input_height * input_width * input_channels * sizeof(float) , cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
          
    pad_input<<<grid1,threads1>>>(pad_input_in, pad_input_out, input_height, input_width, input_channels, padding/2);

    cudaEventRecord(stop);

    err = cudaGetLastError(); 
    if(err!=cudaSuccess) {
      fprintf(stderr, "Failed to launch pad input (error code %s)!\n", cudaGetErrorString(err)); 
      exit(EXIT_FAILURE);
    }
    
    cudaMemcpy(&h_X[i * input_channels * new_input_height * new_input_width], pad_input_out,
              new_input_height * new_input_width * input_channels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    overhead_time += milliseconds;        
  }
  cudaFree(pad_input_in); 
  cudaFree(pad_input_out);

  float *d_X, *d_Y, *d_W; 

  /*allocating memory for padded matrix in the device*/
  err = cudaMalloc((void**)&d_X, size_input_matrix);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector d_X (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  /* copying padded matrix to device */
  err = cudaMemcpy(d_X , h_X , size_input_matrix , cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector h_X from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
 
  /* Renaming the kernel weights pointer (input is in device memory) */
  err = cudaMalloc((void**)&d_W, size_kernel_matrix);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector d_W(error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  /* copying kernel to device */
  err = cudaMemcpy(d_W , weights , size_kernel_matrix , cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector weights from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  /*allocating memory for the output matrix*/
  err = cudaMalloc((void**)&d_Y, size_output_matrix);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector d_Y (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  /* make sure that 1024 threads isn't crossed*/
  int tile_width = 16 , tile_height = 16;   
  int w_grid = ceil((W_out*1.0) / tile_width);
  int h_grid = ceil((H_out*1.0) / tile_height);
 
  int temp  = w_grid * h_grid;
  dim3 grid(batchsize_of_data , out_channels , temp);
  dim3 block(tile_width , tile_height , 1);

  cudaEventRecord(start);

  size_t shmem_size = sizeof(float) * (((stride*tile_width) + kernel_width - stride)*((stride*tile_height) + kernel_height - stride) + kernel_width*kernel_height);
 
  /* calling the direct_convolution kernel */  
  direct_convolution<<< grid, block, shmem_size>>>(input_channels, input_height, input_width, out_channels, kernel_height, kernel_width, 
                                        padding, stride, H_out, W_out, w_grid, tile_width, d_X, d_W, d_Y);
  
  cudaEventRecord(stop);

  err = cudaGetLastError();

  /*checking if the device program is executed or not*/
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch reduce1 kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  /* copying output matrix to host */
  err = cudaMemcpy(h_Y, d_Y, size_output_matrix, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
 
  /* releasing all the device and host vectors */
  err = cudaFree(d_X);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector X (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_Y);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector Y (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_W);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector W (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  /*releasing the memory*/
  free(h_X);

  /*Return the CUDA Array*/
  return h_Y;      
} 
