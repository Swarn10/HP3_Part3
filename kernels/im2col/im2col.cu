#include "im2col.hpp"
#define SHMEM_SIZE 32
__global__ void matrixMul(const float *A,const float *B, float *C, int m, int n, int k, int batch_size) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocated shared memory
    __shared__ float s_a[SHMEM_SIZE];
    __shared__ float s_b[SHMEM_SIZE];
    for(int t = 0;t < batch_size; t++) {
        // Accumulate in temporary variable
        float tmp = 0;
        // Sweep tile across matrix
        for (int i = 0; i < k; i += blockDim.x) {
          // Load in elements for this tile
          s_a[threadIdx.y * blockDim.x + threadIdx.x] = A[row * k + i + threadIdx.x + t*m*k];
          s_b[threadIdx.y * blockDim.x + threadIdx.x] = B[i * n + threadIdx.y * n + col];

          // Wait for both tiles to be loaded in before doing computation
          __syncthreads();

          // Do matrix multiplication on the small matrix
          for (int j = 0; j < blockDim.x; j++) {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
          }

          // Wait for all threads to finish using current tiles before loading in new
          // ones
          __syncthreads();
    }
    
    // Write back results
    C[row * n + col +  t*m*n] = tmp;
  }  
}

// converts a batch of images of shape: data_im: batch x ic x ih x iw (ic: input_channels in image)
// to 2D col of shape: data_col: batch x (ic * kh * kw) x (hcol * wcol)
// filter size: kh x kw
// kernel multiplication patches: hcol x wcol (Based on input size, kernel size, padding, stride)
// Each thread writes one kernel multiplication patch (kh x kw) in data_col
// n is the number of tasks (here: ic * hcol * wcol, ie number of kernel patches per image)
__global__ void im2col_kernel(const float * data_im, float * data_col, const int n,
							  const int kh, const int kw, const int pad, const int stride,
							  const int ih, const int iw, const int ic,
							  const int hcol, const int wcol) 
{
	// esentially this loop could have run batch size number of times
	// but since we are launching enough threads to handle each image separately, it executes just once
	// here it is majorly prevents any extra threads we launch from accessing memory
	CUDA_KERNEL_LOOP(index, n)
	{
		// figure out which part of which image you will work on
		int imidx = blockIdx.y;
		int w_out = index % wcol;
		index /= wcol;
		int h_out = index % hcol;
		int channel_in = index / hcol;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		// this thread will write the output patch (kh x kw) at location (imidx, channel_out, h_out, w_out)
		// that patch is based on the image patch at (imidx, channel_in, h_in, w_in)
		// i.e. will do the work for patch centred at (channel_in, h_in, w_in) in image imidx
		data_im += ((imidx * ic + channel_in) * ih + h_in) * iw + w_in;
		data_col += ((imidx * ic + channel_in) * kh * kw * hcol + h_out) * wcol + w_out;
		#pragma unroll
		for (int i = 0; i < kh; ++i) {
			for (int j = 0; j < kw; ++j) {
				int h = h_in + i;
				int w = w_in + j;
				*data_col = (h >= 0 && w >= 0 && h < ih && w < iw) ?
				  data_im[i * iw + j]: 0;
				data_col += hcol * wcol;
			}
		}
	}
}

// takes a batch of images on GPU: bs x ic x ih x iw (ic: input channels, bs: batch size)
// and the kernels on GPU: oc x ic x kh x kw (oc: output channels)
// does the convolution based on padding (pad) and stride
// data_col is used for intermediate col form storage
// output is returned in data_out
void im2col_gemm_gpu(const float * data_im, const float * data_ker,
					 const int kh, const int kw, const int pad, const int stride,
					 const int ih, const int iw, const int ic, const int oc,
					 float * data_col, float * data_out, int bs)
{
	//cudaDeviceProp devp;
	//cudaGetDeviceProperties(&devp, 0);
	//unsigned int SHMEM_SIZE = devp.sharedMemPerBlock/(2*sizeof(float));
	//printf("Total shared memory per block:%u\n", SHMEM_SIZE);
	// Step 1: convert the image to col form
	
	// dimensions of the col corr to this image
	int hcol = (ih + 2 * pad - kh) / stride + 1;
	int wcol = (iw + 2 * pad - kw) / stride + 1;

	// We are going to launch bs groups of ic * hcol * wcol kernels threads for im2col,
	// each thread is responsible for copying a single-channel kernel multiplication patch
	// i.e. one thread per output pixel in the output of conv
	// So, all images in batch are converted to col form parallely
	int op_size = ic * hcol * wcol;
	dim3 blocks(GET_BLOCKS(op_size), bs, 1);
	dim3 threads(CUDA_NUM_THREADS, 1, 1);
	im2col_kernel<<<blocks, threads>>>(data_im, data_col, op_size, kh, kw, pad, stride, ih, iw, ic, hcol, wcol);
	//CUDA_POST_KERNEL_CHECK; // check if there was any error

	// now, the col form shall be multiplied with the kernels laid out straight i.e. (ic * kh * kw)
	// so, since, oc is the number of kernels, we get:
	// "2D kernel matrix" oc x (ic * kh * kw)
	// and the "2D col matrix" for one image is: (ic * kh * kw) x (hcol * wcol)
	// and you see that magically, their multiplication output is:
	// output: oc x (hcol * wcol)... ie oc x hcol x wcol, the exact shape needed by next convolution
	// output: oc x (hcol * wcol)... ie oc x hcol x wcol, the exact shape needed by next im2col
	// so, there is no need to ever work things back (col2im) or reshape either
	// in sumamary, we do matmul(kernel, im2col(im_input)) -> conv_output (in "correct" form)

	// Step 2: GEMM using libcublas

	// get params ready for GEMM call
	// Performs C + i*strideC = α op(A + i*strideA) op(B + i*strideB) + β(C + i* strideC) 
	// for i ∈ [0, batchSize − 1]
	// Thus, this one call will parallely do the matrix multiplication for all images in the batch
	// Since we are doing A * B, we need α = 1, β = 0
	// Since we don't need any transpose, op = CUBLAS_OP_N
	const float alpha = 1.0f;
	const float beta  = 0.0f;
	int ldA, ldB, ldC;
	int m = ldA = ldC = hcol * wcol;
	int n = oc;
	int k = ldB = ic * kh * kw;
	long long int strideA = m * k;	// size of each col form
	long long int strideB = 0;		// reusing the same kernel matrix for each image
	long long int strideC = m * n;	// size of output feature map
	
	// CUDA sees matrices as column major
	// So, a matrix we see as HxW, it would see as WxH in the same memory layout
	// So, matA (our view) -> matA' (CUDA view)
	// Thus, to do matA * matB in our view, we shall run CUDA for matB * matA.
	// Output would be matB' * matA' (CUDA view) = (matA * matB)' (CUDA view) = matA * matB (our view)
	// In essence, trust me when I do col * kernel to achieve kernel * col

	// Blocks per grid dimension (assumes CUDA_NUM_THREADS divides M and n evenly)
	int BLOCKS_X = n / CUDA_NUM_THREADS;
	int BLOCKS_Y = m / CUDA_NUM_THREADS;
  
	// Use dim3 structs for block  and grid dimensions
	dim3 threads2(CUDA_NUM_THREADS, CUDA_NUM_THREADS);
	dim3 blocks2(BLOCKS_X, BLOCKS_Y);
  
	// Launch kernel
	matrixMul<<<blocks2, threads2>>>(data_col, data_ker, data_out, m, n, k, bs);
}


// takes a batch of images on CPU: data_im:  batch x ic x ih x iw (ic: input channels)
// and the kernels on CPU: data_ker: oc x ic x kh x kw (oc: output channels)
// does the convolution based on padding (pad) and stride
// returns the convolution output on CPU
// conv_time & overhead_time are used for kernel timing
float * im2colWithCuda(const float * data_im, const float * data_ker, const int batch,
					   const int kh, const int kw, const int pad, const int stride,
					   const int ih, const int iw, const int ic, const int oc, 
					   float& conv_time, float& overhead_time)
{
	
	// Timing variables - CUDA Event API
	overhead_time = 0;
	conv_time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);  

	// image dim
	ssize_t image_size = ic * ih * iw;
	ssize_t images_size = batch * image_size;

	// kernel dim
	ssize_t k = ic * kh * kw;
	ssize_t kernels_size = oc * k;

	// col dim
	ssize_t hcol = (ih + 2 * pad - kh) / stride + 1;
	ssize_t wcol = (iw + 2 * pad - kw) / stride + 1;
	ssize_t one_col = ic * kh * kw * hcol * wcol;
	ssize_t col_batch = batch * one_col;

	// output dim
	ssize_t output_feature = oc * hcol * wcol;	
	ssize_t result_size = batch * output_feature;
	
	// move images to GPU
	float * dev_image = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&dev_image, images_size * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(dev_image, data_im, images_size * sizeof(float), cudaMemcpyHostToDevice));
	
	// move kernels to GPU
	float * dev_kernel = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&dev_kernel, kernels_size * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(dev_kernel, data_ker, kernels_size * sizeof(float), cudaMemcpyHostToDevice));

	// allocate GPU memory for intermediate col form
	float * dev_col = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&dev_col, col_batch * sizeof(float)));

	// allocate GPU memory for convlution result
	float * dev_ret = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&dev_ret, result_size * sizeof(float)));
	
	// Record the kernel run time
	cudaEventRecord(start);
	// Kernel launch - this single call will handle all the images in the batch parallely
	im2col_gemm_gpu(dev_image, dev_kernel, kh, kw, pad, stride, ih, iw, ic, oc, dev_col, dev_ret, batch);
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&conv_time, start, stop);
	
	// Check for any errors launching the kernel
	//CUDA_POST_KERNEL_CHECK;

	// Copy output vector from GPU to host memory.
	float * data_ret = (float *)malloc(result_size * sizeof(float));
	CUDA_CHECK(cudaMemcpy(data_ret, dev_ret, result_size * sizeof(float), cudaMemcpyDeviceToHost));

	// Free CUDA memory
	cudaFree(dev_image);
	cudaFree(dev_col);
	cudaFree(dev_kernel);
	cudaFree(dev_ret);
	
	// Free timing resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return data_ret;
}

// The exposed library function which just calls im2colWithCuda the right way
float* IM2COL::forward(int out_size, int channel, int kernel_height, int kernel_width, int pad, 
		int stride, float* kernel, int batch_size, int input_height, int input_width, float* input, 
		float& conv_time, float& overhead_time)
{
	return im2colWithCuda(input, kernel, batch_size, kernel_height, kernel_width, 
					pad, stride, input_height, input_width, channel, out_size, conv_time, overhead_time);
}
