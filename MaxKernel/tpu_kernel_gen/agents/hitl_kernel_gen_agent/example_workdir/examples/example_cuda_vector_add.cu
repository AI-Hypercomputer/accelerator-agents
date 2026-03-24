/**
 * Example CUDA kernel for vector addition
 * This demonstrates a simple GPU kernel that should be converted to JAX
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add_kernel(float* A, float* B, float* C, int N) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void launch_vector_add(float* d_A, float* d_B, float* d_C, int N) {
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    vector_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);

    // Synchronize
    cudaDeviceSynchronize();
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    launch_vector_add(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
