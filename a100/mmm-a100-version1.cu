#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16 // Define the size of the blocks

// CUDA Kernel for matrix multiplication (C = A * B)
__global__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // The element of the block sub-matrix that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        // Shared memory for the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Shared memory for the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory to shared memory
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

int main() {
    // Size of matrices
    int N = 20480; // For simplicity, assuming square matrices
    int size = N * N * sizeof(float);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // Device pointers

    // Allocate memory on host
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Initialize matrices on host
    for (int i = 0; i < N * N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy host memory to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // Launch the CUDA Kernel
    matrixMulCUDA<<<grid, threads>>>(d_c, d_a, d_b, N, N);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}