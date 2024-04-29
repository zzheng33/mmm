#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32 // Larger block size to better utilize cache

// CUDA Kernel for matrix multiplication (C = A * B)
__global__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the row index of the C element to work on
    int row = by * BLOCK_SIZE + ty;

    // Calculate the column index of the C element to work on
    int col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0;

    // Loop over the A and B matrices to compute the dot product
    for (int m = 0; m < (wA - 1) / BLOCK_SIZE + 1; ++m) {
        // Collaboration loading of A and B into shared memory
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int A_col = m * BLOCK_SIZE + tx;
        int B_row = m * BLOCK_SIZE + ty;
        if (A_col < wA && row < wA) {
            As[ty][tx] = A[row * wA + A_col];
        } else {
            As[ty][tx] = 0.0;
        }

        if (B_row < wB && col < wB) {
            Bs[ty][tx] = B[B_row * wB + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        // Compute the dot product for the C element
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < wA && col < wB) {
        C[row * wB + col] = Cvalue;
    }
}

int main() {
    // Matrix dimensions
    int N = 20480; // Assuming square matrices for simplicity
    int size = N * N * sizeof(float);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Transfer data to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Execution configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Execute the kernel
    matrixMulCUDA<<<grid, threads>>>(d_c, d_a, d_b, N, N);

    // Copy results back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
