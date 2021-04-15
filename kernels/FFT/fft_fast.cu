// Shubhankar_Banerjee 18EC10056
// Siddharth Gupta 18EC10057

                                                              // CONVOLUTION USING FFT 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <assert.h>
#include "fftheader.h"

  /* General Error Checking Code */
static const char * _cudaGetErrorEnum(cufftResult error) 
{
switch (error)
{
case CUFFT_SUCCESS:
  return "CUFFT_SUCCESS";
case CUFFT_INVALID_PLAN:
  return "The plan parameter is not a valid handle";

case CUFFT_ALLOC_FAILED:
  return "The allocation of GPU or CPU memory for the plan failed";

case CUFFT_INVALID_TYPE:
  return "CUFFT_INVALID_TYPE";

case CUFFT_INVALID_VALUE:
  return "One or more invalid parameters were passed to the API";

case CUFFT_INTERNAL_ERROR:
  return "An internal driver error was detected";

case CUFFT_EXEC_FAILED:
  return "cuFFT failed to execute the transform on the GPU";

case CUFFT_SETUP_FAILED:
  return "The cuFFT library failed to initialize";

case CUFFT_INVALID_SIZE:
  return "One or more of the parameters is not a supported size";

case CUFFT_UNALIGNED_DATA:
  return "CUFFT_UNALIGNED_DATA";

case CUFFT_INCOMPLETE_PARAMETER_LIST:
  return "Missing parameters in call";

case CUFFT_INVALID_DEVICE : 
  return "An invalid GPU index was specified in a descriptor or Execution of a plan was on different GPU than plan creation";

case CUFFT_PARSE_ERROR : 
  return "Internal plan database error";

case CUFFT_NO_WORKSPACE :  
  return "No workspace has been provided prior to plan execution";

case CUFFT_NOT_IMPLEMENTED : 
  return "Function does not implement functionality for parameters given";

case CUFFT_LICENSE_ERROR :
  return "Used in previous versions";

case CUFFT_NOT_SUPPORTED : 
  return "Operation is not supported for parameters given";
}

return "<unknown>";
}

#define cufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
if (CUFFT_SUCCESS != err)
{
  fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n", __FILE__, __LINE__, err,
          _cudaGetErrorEnum(err));
  cudaDeviceReset();
  assert(0);
}
}


/*Central element of the old_filter in the (0,0,0) position of the new_filter.
 *(x,y,z) -> ((x-X/2)%X, (y-Y/2)%Y, (z-Z/2)%Z)
 *new_filter[RHS] = old_filter[LHS]
 */
__global__ void align_filter(float *align_inp, float *align_output, int H, int W, int D, int out_size)
{
  //allocation of thread ids in all dimensions
  int coloumn = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int depth = blockIdx.z * blockDim.z + threadIdx.z;

  int new_coloumn = ((coloumn - H / 2) % H);  
  int new_row = ((row - W / 2) % W);
  int new_depth = ((depth - D / 2) % D);

  if (new_coloumn < 0)
      new_coloumn = H + new_coloumn;
  if (new_row < 0)
      new_row = W + new_row;
  if (new_depth < 0)
      new_depth = D + new_depth;

  if (coloumn < H && row < W && depth < D)
  {
      #pragma unroll
      for (int it = 0; it < out_size; it++)
      {
          int i = it * D * H * W + depth * H * W + coloumn * W + row;
          int j = it * D * H * W + new_depth * H * W + new_coloumn * W + new_row;
          align_output[j] = align_inp[i];
      }
  }
}

/*flip filter about the center element */
__global__ void flip_filter(float *flip_inp, float *flip_output, int k_len, int k_width, int k_height, int out_size)
{
  //allocation of thread ids in all dimensions 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int coloumn = blockIdx.x * blockDim.x + threadIdx.x;
  int depth = blockIdx.z * blockDim.z + threadIdx.z;

  int new_coloumn = k_len - coloumn - 1;   //new coloumn index i->n-i-1
  int new_row = k_width - row - 1;
  int new_depth = k_height - depth - 1;

  if (coloumn < k_len && row < k_width && depth < k_height)
  {
      #pragma unroll
      for (int itr = 0; itr < out_size; itr++)
      {
          int i = itr * k_height * k_len * k_width + depth * k_len * k_width + coloumn * k_width + row;
          int j = itr * k_height * k_len * k_width + new_depth * k_len * k_width+ new_coloumn * k_width + new_row;
          flip_output[j] = flip_inp[i];
      }
  }
}

// PADDING 
__global__ void do_pad(float *pad_input, float *pad_output, int len, int width, int height, int pad_front, int pad_back, int batch_size)
{
//allocation of thread ids in all dimensions 
 
int coloumn = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int depth = blockIdx.z * blockDim.z + threadIdx.z;

int new_pad_len = len + pad_front + pad_back;
int new_pad_width = width + pad_front + pad_back;

if (coloumn < new_pad_len && row < new_pad_width && depth < height)
{
  #pragma unroll
 //iterate over the batch_size and provide padded output
  for (int it = 0; it < batch_size; it++)
  {
    int i = it * height * new_pad_len * new_pad_width + depth * new_pad_len * new_pad_width + coloumn * new_pad_width + row;
    int j = it * height * len * width + depth * len * width + (coloumn - pad_front) * width + (row - pad_front);

    if ((coloumn < pad_front || coloumn > len + pad_back - 1) || (row < pad_front || row > width + pad_back - 1))
      pad_output[i] = 0;
    else
      pad_output[i] = pad_input[j];
  }
}
}

// INPUT IMAGE FFT
cufftComplex *compute_fft_input(float *input_layer, int pad, int batchsize, int *il_dim, float &conv_time, float &overhead_time)
{
cudaError_t err = cudaSuccess; // check error
int len = il_dim[0];
int width = il_dim[1];
int height = il_dim[2];

// Profiling
float milliseconds = 0;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

/* pad input */
int pad_len = len + 2 * pad;
int pad_width = width + 2 * pad;

// padding input
float *pad_ilayer = NULL;
cudaMalloc((void **)&pad_ilayer, batchsize * len * width * height * sizeof(float));
cudaMemcpy(pad_ilayer, input_layer, batchsize * len * width * height * sizeof(float), cudaMemcpyHostToDevice);

// padding output
float *pad_olayer = NULL;
cudaMalloc((void **)&pad_olayer, batchsize * pad_len * pad_width * height * sizeof(float));

dim3 threadsize1(8, 8, 8);
dim3 gridsize1(ceil(pad_len / 8.0f), ceil(pad_width / 8.0f), ceil(height / 8.0f));

int padsize = pad;
cudaEventRecord(start);
do_pad<<<gridsize1, threadsize1>>>(pad_ilayer, pad_olayer, len, width, height, padsize,padsize, batchsize);
cudaEventRecord(stop);

//error
err = cudaGetLastError();
if (err != cudaSuccess)
{
  fprintf(stderr, "Failed to launch pad input (error code %s)!\n", cudaGetErrorString(err));
  exit(EXIT_FAILURE);
}
// Calc overhead time
cudaEventSynchronize(stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
overhead_time += milliseconds;
printf("Time taken for Input_padding : %f\n",milliseconds);

// free  memory
cudaFree(pad_ilayer);

len = pad_len;
width = pad_width;
//input for plan many function
int N[3] = {height, len, width};

cufftComplex *d_input_complex;
// For cufftPlan many
cufftHandle forwardplan_inp;

size_t complex_size = batchsize * height * width * (len / 2 + 1) * sizeof(cufftComplex);

cudaMalloc((void **)&d_input_complex, complex_size);

// Plan function
cudaEventRecord(start);
cufftSafeCall(cufftPlanMany(&forwardplan_inp, 3, N, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, batchsize));
cudaEventRecord(stop);

cudaEventSynchronize(stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
conv_time += milliseconds;
// plan end

/* Execution function start */
cudaEventRecord(start);
cufftSafeCall(cufftExecR2C(forwardplan_inp, pad_olayer, d_input_complex));  //pad_olayer is the padded input which goes cufftexecr2C function
cudaEventRecord(stop);

cudaEventSynchronize(stop);
milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
conv_time += milliseconds;
/* Execution function end */

cudaFree(pad_olayer);
cufftDestroy(forwardplan_inp);
return d_input_complex;
}


// Kernel FFT

cufftComplex *compute_kernel_fft(float *kernel, int pad, int *il_dim, int *kernel_dim, int out_size, float &conv_time, float &overhead_time)
{
  cudaError_t err = cudaSuccess; // check error
  //unrolling the inputs 
  int len = il_dim[0];
  int width = il_dim[1];
  int height = il_dim[2];
  int k_len = kernel_dim[0];
  int k_width = kernel_dim[1];
  int k_height = kernel_dim[2];

//after padding length of input
  int new_len = len + 2 * pad;
  int new_width = width + 2 * pad;
  len = new_len;
  width = new_width;

// Profiling /Time calc:
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*flip filter input output declaration */
  float *flip_inp = NULL;
  cudaMalloc((void **)&flip_inp, out_size * k_len * k_width * k_height * sizeof(float));
  float *flip_output = NULL;
  cudaMalloc((void **)&flip_output, out_size * k_len * k_width * k_height * sizeof(float));
  
  //flip_inp= kernel
  cudaMemcpy(flip_inp, kernel, out_size * k_len * k_width * k_height * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsize(8, 8, 8);
  dim3 gridsize(ceil(k_len / 8.0f), ceil(k_width / 8.0f), ceil(k_height / 8.0f));

  cudaEventRecord(start);
  flip_filter<<<gridsize, threadsize>>>(flip_inp, flip_output, k_len, k_width, k_height, out_size);
  cudaEventRecord(stop);
 
 // error check 
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaFree(flip_inp); 
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
  printf("Time taken for Flip_Filter execution : %f\n",milliseconds);


  /* flip filter end */

  /* pad filter */
  //pad_size determination for kernel making kernel equal to padded input size

  int paalign_outputack = (new_len - k_len) / 2;
  int pad_front;
  if ((new_len - k_len) % 2 == 0)
      pad_front = paalign_outputack;
  else
      pad_front = paalign_outputack + 1;
  int new_k_len = k_len + pad_front + paalign_outputack;
  int new_k_width = k_width + pad_front + paalign_outputack;

  //padding inputs declarations
  float *pad_filter_in = NULL;
  pad_filter_in = flip_output;

  float *pad_filter_out = NULL;
  cudaMalloc((void **)&pad_filter_out, out_size * new_k_len * new_k_width * height * sizeof(float));

// for padding 
  dim3 threadsize1(8, 8, 8);
  dim3 gridsize1(ceil(new_k_len / 8.0f), ceil(new_k_width / 8.0f), ceil(height / 8.0f));
 
 

  cudaEventRecord(start);
  do_pad<<<gridsize1, threadsize1>>>(pad_filter_in, pad_filter_out, k_len, k_width, height, pad_front, paalign_outputack, out_size);
  cudaEventRecord(stop);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch pad kernel_filter(error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  k_len = new_k_len;
  k_width = new_k_width;
  cudaFree(pad_filter_in);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
     printf("Time taken for Filter_padding : %f\n",milliseconds);

  /* pad filter end */

  // align filter 
  float *align_inp = NULL;
  align_inp = pad_filter_out;
  float *align_output = NULL;
  cudaMalloc((void **)&align_output, out_size * k_len * k_width * k_height * sizeof(float));
  

  dim3 threads3(8, 8, 8);
  dim3 grid3(ceil(k_len / 8.0f), ceil(k_width / 8.0f), ceil(k_height /8.0f));

  cudaEventRecord(start);
  align_filter<<<grid3, threads3>>>(align_inp, align_output, k_len, k_width, k_height, out_size);
  cudaEventRecord(stop);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch align_filter(error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  cudaFree(align_inp);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;
  printf("Time taken for Filter_aligning : %f\n",milliseconds);

  /* align filter end */

  int N[3] = {height, len, width};
  cufftComplex *kernel_fft;
  cufftHandle k_widthplan_input;
  
  size_t complex_size = (out_size+1) * height * width * (len / 2 + 1) * sizeof(cufftComplex);

  cudaMalloc((void **)&kernel_fft, complex_size);
  cudaMemset(kernel_fft, 0, complex_size);

  cudaEventRecord(start);
  cufftSafeCall(cufftPlanMany(&k_widthplan_input, 3, N, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, out_size));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  cudaEventRecord(start);
  cufftSafeCall(cufftExecR2C(k_widthplan_input, align_output, kernel_fft));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  conv_time += milliseconds;

  cudaFree(align_output);
  cufftDestroy(k_widthplan_input);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return kernel_fft;
}

__global__ void pointwise_product( float len, float scale_factor,cufftComplex *data_outA, cufftComplex *data_outB)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < len)
{
  float m, n;
  m = data_outA[i].x * data_outB[i].x - data_outA[i].y * data_outB[i].y;
  n = data_outA[i].x * data_outB[i].y + data_outA[i].y * data_outB[i].x;
  data_outA[i].x = scale_factor * m  ;
  data_outA[i].y = scale_factor * n ;
}
}


__global__ void crop_with_stride(float *f_out, int H, int W, int nos_oHeight, int nos_oWidth, int D, int stride, int out_len,float *f_in)
{
int r = blockIdx.y * blockDim.y + threadIdx.y;  
int c = blockIdx.x * blockDim.x + threadIdx.x;
int batch = blockIdx.z * blockDim.z + threadIdx.z;
int i = (((D - 1) / 2) * H * W + c * W) + r + (batch * D * H * W) ;

int crop_Height_1 = (H - nos_oHeight) / 2;
int crop_Height_2;int crop_Width_2;
int crop_Width_1 = (W - nos_oWidth) / 2;
 

 
if ((H - nos_oHeight) % 2 == 0)
  crop_Height_2 = crop_Height_1;
else
  crop_Height_2 = crop_Height_1 + 1;
if ((W - nos_oWidth) % 2 == 0)
  crop_Width_2 = crop_Width_1;
else
  crop_Width_2 = crop_Width_1 + 1;

int j = batch * nos_oHeight * nos_oWidth + (c - crop_Height_2) * nos_oWidth + (r - crop_Width_2);

if ((r < W)  && (c < H) && (batch < out_len))
{
  if ((c >= crop_Height_2) && (r < W - crop_Width_1) && (c < H - crop_Height_1) && (r >= crop_Width_2))
  {
    if (stride == 1)
      f_out[j] = f_in[i];
    else
    {
      if (((c - crop_Height_2) % stride) == 0 && ((r - crop_Width_2) % stride == 0))
      {
        j = batch * (nos_oHeight / stride + 1) * (nos_oWidth / stride + 1) + (((c - crop_Height_2) / stride) * (nos_oWidth / stride + 1)) + ((r - crop_Width_2) / stride);
        f_out[j] = f_in[i];
      }
    }
  }
}
}

__global__ void copy_ip(int len, int H, int W, int D,cufftComplex *img_fft, cufftComplex *data_outA)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < len)
{
  data_outA[i] = img_fft[i % (D * ((H/2) + 1) * W)];
}
}


float *conv_op(int Height, int Width, int Depth, int Out_Size,cufftComplex *kernel_fft, cufftComplex *img_fft, float &conv_time, float &overhead_time)
{
float ms = 0;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

int new_Out_S = Out_Size + 1;
int dim_arr[3] = {Depth, Height, Width};
cufftReal *data_inA;
cufftComplex *data_outA, *data_outB;
cufftHandle inv_fft_plan;
size_t R_size = new_Out_S * Depth * Width * Height * sizeof(cufftReal);
size_t C_size = new_Out_S * Depth * Height * (Width / 2 + 1) * sizeof(cufftComplex);

cudaMalloc((void **)&data_outA, C_size);
cudaMalloc((void **)&data_inA, R_size);
cudaMemset(data_inA, 0, R_size);


int blocks_num = ceil((new_Out_S * Depth * (Height/ 2 + 1) * Width) / 1024.0f);
dim3 t_copy(1024);
dim3 grid4copy(blocks_num);

cudaEventRecord(start);
copy_ip<<<grid4copy, t_copy>>>((new_Out_S * Depth * (Height / 2 + 1) * Width), Height, Width, Depth,img_fft, data_outA);
cudaEventRecord(stop);
 
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
ms = 0;
overhead_time += ms;



/* Using fft library to make a plan for the the inverse transforms */
cudaEventRecord(start);
cufftSafeCall(cufftPlanMany(&inv_fft_plan, 3, dim_arr, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, new_Out_S));
cudaEventRecord(stop);

cudaEventSynchronize(stop);
ms = 0;
cudaEventElapsedTime(&ms, start, stop);
conv_time += ms;
data_outB = kernel_fft;

int blocks_number = ceil((new_Out_S * Depth* Height * (Width/ 2 + 1)) / 1024.0f);
dim3 thread_pws(1024);
dim3 grid_pws(blocks_number);

cudaEventRecord(start);
pointwise_product<<<grid_pws, thread_pws>>>((new_Out_S * Depth * Height * (Width / 2 + 1)), 1.0f / (Height * Width * Depth),data_outA, data_outB);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
ms = 0;
cudaEventElapsedTime(&ms, start, stop);
conv_time += ms;

/* Inverse FFT of output using the cufftExec function*/
cudaEventRecord(start);
cufftSafeCall(cufftExecC2R(inv_fft_plan, data_outA, data_inA));
cudaEventRecord(stop);

cudaEventSynchronize(stop);
ms = 0;
cudaEventElapsedTime(&ms, start, stop);
conv_time += ms;

/* Releasing the used memory */
cudaFree(data_outA);
cufftDestroy(inv_fft_plan);
cudaEventDestroy(start);
cudaEventDestroy(stop);

return data_inA; //inverse of fft multiplication returned 
}


//multiplying FFTs
float* pointwise_multiply_FFTs(cufftComplex* img_fft, cufftComplex* kernel_fft, int pad, int stride, int batch_size, int* il_dim, int* ker_dimen, int out_size,
                                                                            float& conv_time, float& overhead_time) 
{
float ms = 0;
int Height = il_dim[0]; 
int Width = il_dim[1]; 
int Depth = il_dim[2]; 
int k_H = ker_dimen[0];
int  k_W = ker_dimen[1];
cudaError_t err = cudaSuccess;
  
int new_H = Height+2*pad; int new_W = Width+2*pad;
Height = new_H; 
Width = new_W;
int b_padding = (new_H - k_H)/2;
int f_padding; 
if((new_H - k_H) % 2 == 0) 
      f_padding = b_padding; 
else 
    f_padding = b_padding + 1;  
 
 /* making the dimensions of the o/p correct*/
int new_fH = k_H+f_padding+b_padding; 
int new_fW = k_W+f_padding+b_padding;
k_H = new_fH; k_W = new_fW;


cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

/* Doing pointwise multiplication of ffts */
float* mul_result = conv_op(Height, Width, Depth, out_size,kernel_fft,img_fft, conv_time, overhead_time);

/* cropping the  output */
k_H = ker_dimen[0]; k_W = ker_dimen[1] ;
int out_Height = (Height - k_H)/stride + 1; 
int out_Width = (Width - k_W)/stride + 1;
int nos_oHeight = (Height - k_H + 1); 
int nos_oWidth = Width -k_W + 1;
float* result_final = (float*)malloc((out_size) * out_Width*out_Height* sizeof(float));
float *crop_out = NULL; 
err = cudaMalloc((void **)&crop_out, out_size * out_Height * out_Width * sizeof(float));
if(err!=cudaSuccess)
{
    fprintf(stderr, "Failed to allocate memory crop_out (error code %s)!\n", cudaGetErrorString(err)); 
    exit(EXIT_FAILURE);
}
float *crop_in = NULL; 
crop_in = mul_result; 

dim3 threads_crop(8,8,8);
dim3 grid_crop(ceil(Height/8.0f),ceil(Width/8.0f),ceil(out_size/8.0f));

cudaEventRecord(start);
crop_with_stride<<<grid_crop, threads_crop>>>( crop_out, Height, Width, nos_oHeight, nos_oWidth, Depth, stride, out_size,crop_in);
cudaEventRecord(stop);


err = cudaGetLastError();
if(err!=cudaSuccess)
{
    fprintf(stderr, "Failed to launch crop(error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

//copying the final result from the device memory to the host memory
cudaMemcpy(result_final, crop_out, out_size* out_Width*out_Height* sizeof(float) ,cudaMemcpyDeviceToHost);
cudaEventSynchronize(stop);
ms = 0;
cudaEventElapsedTime(&ms, start, stop);
overhead_time += ms;
printf("Output_crop&stride_time: %f\n",ms);

cudaFree(crop_in); 
cudaFree(crop_out);
/* crop output end */

cudaEventDestroy(start);
cudaEventDestroy(stop);
return result_final;
}


/* Implementation of the forward pass of FFT Kernel */
float* FFT::forward(int out_size, int channel, int kernel_len, int kernel_width, int pad, int stride, float *kernel,
             int batch_size, int len, int width, float *input_layer_img, float &conv_time, float &overhead_time)
{
int il_dim[3] = {len, width, channel};
int kernel_dim[3] = {kernel_len, kernel_width, channel};


// Initialising the time to be calculated
conv_time = 0;
overhead_time = 0;
float milliseconds = 0;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

int H =len + 2 * pad;
int W = width + 2 * pad;
int out_H = ((H -kernel_len ) / stride) + 1;       //number of elements in output length
int out_W = ((W - kernel_width) / stride) + 1;       //number of elements in output width

cufftComplex *input_fft = compute_fft_input(input_layer_img, pad, batch_size, il_dim, conv_time, overhead_time);        // input_image fft stored as cufft complex 1D array
cufftComplex *kernel_fft = compute_kernel_fft(kernel, pad, il_dim, kernel_dim, out_size, conv_time, overhead_time);       //kernel fft stored as cufft complex  1Darray

//final output of convolution using fft
float *final_output = (float *)malloc(batch_size * out_size * out_H * out_W * sizeof(float));  // dimensions l*w*(number of 3D kernels/filters used)*(batch_size of input)

for (int l = 0; l < batch_size; l++)
{
  // convolution using fft result 
  float *actual_result = pointwise_multiply_FFTs(&input_fft[l * channel * (H / 2 + 1) * W], kernel_fft, pad, stride, batch_size, il_dim, kernel_dim,
                                      out_size, conv_time, overhead_time);

  cudaEventRecord(start);
  #pragma unroll
  for (int itr1 = 0; itr1 < out_size; itr1++)
  {
    for (int itr2 = 0; itr2 < out_H; itr2++)
    {
      for (int itr3 = 0; itr3 < out_W; itr3++)
      {
        final_output[l * out_size * out_H * out_W + itr1 * out_H * out_W + itr2 * out_W + itr3] = actual_result[itr1 * out_H * out_W + itr2 * out_W + itr3]; //accumulating all batches results into one single final array
      }
    }
  }

  cudaEventRecord(stop);
// adding to the resultant time
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  overhead_time += milliseconds;

  free(actual_result);
 
}
cudaFree(input_fft);
cudaFree(kernel_fft);
cudaEventDestroy(start);
cudaEventDestroy(stop);
return final_output;
 

 }
/* Main 
int main()
{
int channel = 3;
int height = 5;
int width = 5;
int kernel_height = 3;
int kernel_width = 3;
int batch_size = 3;
int pad = 1;
int stride = 2;
int out_size = 2;
float input_layer_tmp[batch_size][channel][height][width] = {
    {{{0, 0, 1, 0, 2}, {1, 0, 2, 0, 1}, {1, 0, 2, 2, 0}, {2, 0, 0, 2, 0}, {2, 1, 2, 2, 0}},
     {{2, 1, 2, 1, 1}, {2, 1, 2, 0, 1}, {0, 2, 1, 0, 1}, {1, 2, 2, 2, 2}, {0, 1, 2, 0, 1}},
     {{2, 4, 4, 2, 0}, {1, 0, 0, 1, 0}, {0, 1, 0, 0, 0}, {1, 0, 2, 1, 0}, {2, 2, 1, 1, 1}}},
    {{{0, 0, 1, 0, 2}, {1, 0, 2, 0, 1}, {1, 0, 2, 2, 0}, {2, 0, 0, 2, 0}, {2, 1, 2, 2, 0}},
     {{2, 1, 2, 1, 1}, {2, 1, 2, 0, 1}, {0, 2, 1, 0, 1}, {1, 2, 2, 2, 2}, {0, 1, 2, 0, 1}},
     {{2, 1, 1, 2, 0}, {1, 0, 0, 1, 0}, {0, 1, 0, 0, 0}, {1, 0, 2, 1, 0}, {2, 2, 1, 1, 1}}},

    {{{0, 0, 1, 0, 2}, {1, 0, 2, 0, 1}, {1, 0, 2, 2, 0}, {2, 0, 0, 2, 0}, {2, 1, 2, 2, 0}},
     {{2, 1, 2, 1, 1}, {2, 1, 2, 0, 1}, {0, 2, 1, 0, 1}, {1, 2, 2, 2, 2}, {0, 1, 2, 0, 1}},
     {{2, 1, 1, 2, 0}, {1, 0, 0, 1, 0}, {0, 1, 0, 0, 0}, {1, 0, 2, 1, 0}, {2, 2, 1, 1, 1}}}

};

float kernel_tmp[out_size][channel][kernel_height][kernel_width] =
    {
        {{{-1, 0, 1}, {0, 0, 1}, {1, -1, 1}},
         {{-1, 0, 1}, {1, -1, 1}, {0, 1, 0}},
         {{-1, 1, 1}, {1, 1, 0}, {0, -1, 0}}},
        {{{-1, 0, 1}, {0, 0, 1}, {1, -1, 1}},
         {{-1, 0, 1}, {1, 0, 1}, {0, 1, 0}},
         {{-1, 1, 1}, {1, 1, 0}, {0, -1, 0}}}};

float *input_layer = (float *)malloc(batch_size * channel * height * width * sizeof(float));
float *kernel = (float *)malloc(out_size * channel * kernel_height * kernel_width * sizeof(float));
int out_H = ((height - kernel_height + 2 * pad) / stride) + 1;
int out_W = ((width - kernel_width + 2 * pad) / stride) + 1;

float *input_layer_cuda = NULL;
cudaMalloc((void **)&input_layer_cuda, batch_size * channel * height * width * sizeof(float));
float *kernel_cuda = NULL;
cudaMalloc((void **)&kernel_cuda, out_size * channel * kernel_height * kernel_width * sizeof(float));
cudaMemcpy(input_layer_cuda, input_layer_tmp, batch_size * channel * height * width * sizeof(float), cudaMemcpyHostToDevice); 
cudaMemcpy(kernel_cuda, kernel_tmp, out_size * channel * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);

float a;
float b;
float *a1 = &a;
float *b1 = &b;

float *final_out = forward(out_size, channel, kernel_height, kernel_width, pad, stride, kernel_cuda, batch_size, height, width, input_layer_cuda, a, b);
for (int l = 0; l < batch_size; l++)
{
  for (int i = 0; i < out_size; i++)
  {
    for (int j = 0; j < out_H; j++)
    {
      for (int k = 0; k < out_W; k++)
      {
          //final_temp[l][i][j][k] = final_output[l * out_size * out_H * out_W + i * out_H * out_W + j * out_W + k]);
       printf("%f ", final_output[l * out_size * out_H * out_W + i * out_H * out_W + j * out_W + k]);
       // printf("%f ",final_temp[l][i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n\n");
}
return 0;
} */