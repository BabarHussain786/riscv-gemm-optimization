//// fork, from recursive_rvv_matmul.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DEFAULT_THRESHOLD 64 

// Recursive matrix multiplication:
// C = C + A * B
// A, B, C point to submatrices in full matrices
// size = dimension of submatrix (power of two assumed)
// stride = full matrix width (for row indexing)
// threshold := threshold for base case (by value, for dynamic setting, CONSTANT in all time execution)
void recursive_multiply(double* A, double* B, double* C, int size, int stride, int threshold) {
    if (size <= threshold) {
    
        //base case, iterate over the tile with nested-reordered loop
        for( int i = 0; i < size; i++ ){
            for( int k = 0; k < size; k++ ){
                for( int j = 0; j < size; j++ ){
                   /// C[i][j] += A[i][k] * B[k][j] 
                   C[ i * stride + j ] += A[ i * stride + k ] * B[ k * stride + j ];
                }
            }
        }
        return;
        
    }

    int new_size = size / 2;
    
    /*
    LEICERSON STEPS (pseudocode)
    
    <task> C00 += A00 * B00
    <task> C01 += A00 * B01
    <task> C10 += A10 * B00
    <task> C11 += A10 * B01
    <sync>
    <task> C00 += A01 * B10
    <task> C01 += A01 * B11
    <task> C10 += A11 * B10
    <task> C11 += A11 * B11
    <sync>
    
    */

    #pragma omp parallel
    {

    // C00 += A00*B00
    #pragma omp task
    recursive_multiply(A, B, C, new_size, stride, threshold);
    
    // C01 += A00*B01
    #pragma omp task
    recursive_multiply(A, B + new_size, C + new_size, new_size, stride, threshold);
    
    // C10 += A10*B00
    #pragma omp task
    recursive_multiply(A + new_size * stride, B, C + new_size * stride, new_size, stride, threshold);
    
    // C11 = A10*B01 
    #pragma omp task
    recursive_multiply(A + new_size * stride, B + new_size, C + new_size * stride + new_size, new_size, stride, threshold);
     
    
    // (sync)
    #pragma omp taskwait 
    
    
    // C00 += A01*B10
    #pragma omp task
    recursive_multiply(A + new_size, B + new_size * stride, C, new_size, stride, threshold);

    // C01 += A01*B11
    #pragma omp task
    recursive_multiply(A + new_size, B + new_size * stride + new_size, C + new_size, new_size, stride, threshold);

    // C10 += A11*B10
    #pragma omp task
    recursive_multiply(A + new_size * stride + new_size, B + new_size * stride, C + new_size * stride, new_size, stride, threshold);

    // C11 += A11*B11
    #pragma omp task
    recursive_multiply(A + new_size * stride + new_size, B + new_size * stride + new_size, C + new_size * stride + new_size, new_size, stride, threshold);

    // (sync)
    #pragma omp taskwait 
    
    }

}

void zero_matrix(double* M, int size) {
    memset(M, 0, size * size * sizeof(double));
}

int main(int argc, char* argv[]) {
    int size = 512; // Default matrix size
    int threshold = DEFAULT_THRESHOLD;

    if (argc == 2) {
        size = atoi(argv[1]);
    }
    else if( argc == 3 ){
        size = atoi(argv[1]);
        threshold = atoi(argv[2]);
    }

    if (size <= 0 || (size & (size - 1)) != 0) {
        fprintf(stderr, "Error: size must be a positive power of two.\n");
        return 1;
    }
    
    printf("size:%d  threshold:%d\n", size, threshold);

    double* A = malloc(size * size * sizeof(double));
    double* B = malloc(size * size * sizeof(double));
    double* C = malloc(size * size * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    srand(1);
    for (int i = 0; i < size * size; i++) {
        A[i] = (double)(rand() % 10 + 1);
        B[i] = (double)(rand() % 10 + 1);
    }

    zero_matrix(C, size);

    double start_time = omp_get_wtime();
    recursive_multiply(A, B, C, size, size, threshold);
    double end_time = omp_get_wtime();

    printf("  <name_version, time[sec], size, threshold>\n");
    printf("> BENCHMARK_RECORD : smatmul_recursive %f, %d, %d\n", end_time - start_time, size, threshold);

    free(A);
    free(B);
    free(C);

    return 0;
}

