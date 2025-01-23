ct #include<stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

    __global__ void computeVectors(int *d_arr1, int *d_arr2, double *d_dotProduct, double *d_length1, double *d_length2, int elementsPerThread)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double dot = 0, len1 = 0, len2 = 0;

    for (int i = 0; i < elementsPerThread; i++)
    {
        int index = idx * elementsPerThread + i;
        dot += d_arr1[index] * d_arr2[index];
        len1 += d_arr1[index] * d_arr1[index];
        len2 += d_arr2[index] * d_arr2[index];
    }

    d_dotProduct[idx] = dot;
    d_length1[idx] = len1;
    d_length2[idx] = len2;
}

void generateRandomArray(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % 10000;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <array_size> <threads_per_block>\n", argv[0]);
        return -1;
    }

    int arraySize = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int blocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    int elementsPerThread = arraySize / (threadsPerBlock * blocks);

    // Host memory allocation
    int *h_arr1 = (int *)malloc(arraySize * sizeof(int));
    int *h_arr2 = (int *)malloc(arraySize * sizeof(int));
    double *h_dotProduct, *h_length1, *h_length2;
    h_dotProduct = (double *)malloc(blocks * threadsPerBlock * sizeof(double));
    h_length1 = (double *)malloc(blocks * threadsPerBlock * sizeof(double));
    h_length2 = (double *)malloc(blocks * threadsPerBlock * sizeof(double));

    // Initialize arrays
    srand(time(NULL));
    generateRandomArray(h_arr1, arraySize);
    generateRandomArray(h_arr2, arraySize);

    // Device memory allocation
    int *d_arr1, *d_arr2;
    double *d_dotProduct, *d_length1, *d_length2;
    cudaMalloc((void **)&d_arr1, arraySize * sizeof(int));
    cudaMalloc((void **)&d_arr2, arraySize * sizeof(int));
    cudaMalloc((void **)&d_dotProduct, blocks * threadsPerBlock * sizeof(double));
    cudaMalloc((void **)&d_length1, blocks * threadsPerBlock * sizeof(double));
    cudaMalloc((void **)&d_length2, blocks * threadsPerBlock * sizeof(double));

    // Transfer data to device
    cudaMemcpy(d_arr1, h_arr1, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch
    computeVectors<<<blocks, threadsPerBlock>>>(d_arr1, d_arr2, d_dotProduct, d_length1, d_length2, elementsPerThread);
    cudaDeviceSynchronize();

    // Transfer results back to host
    cudaMemcpy(h_dotProduct, d_dotProduct, blocks * threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_length1, d_length1, blocks * threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_length2, d_length2, blocks * threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);

    // Combine results on host
    double dotProduct = 0, length1 = 0, length2 = 0;
    for (int i = 0; i < blocks * threadsPerBlock; i++)
    {
        dotProduct += h_dotProduct[i];
        length1 += h_length1[i];
        length2 += h_length2[i];
    }

    length1 = sqrt(length1);
    length2 = sqrt(length2);
    double cosine = dotProduct / (length1 * length2);
    double angle = acos(cosine) * 180.0 / M_PI;

    // Output results
    printf("Dot product: %.2f\n", dotProduct);
    printf("Cosine of angle: %.4f\n", cosine);
    printf("Angle (degrees): %.2f\n", angle);

    // Free memory
    free(h_arr1);
    free(h_arr2);
    free(h_dotProduct);
    free(h_length1);
    free(h_length2);
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_dotProduct);
    cudaFree(d_length1);
    cudaFree(d_length2);

    return 0;
}
