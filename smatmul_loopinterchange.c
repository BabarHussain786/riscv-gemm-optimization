#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IFDEBUG if(DEBUG_FLAG)

#define DEBUG_FLAG 0
#define SIZE 2


// Function to perform matrix multiplication
void multiply(double* mat1, double* mat2, double* res, int size) {
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            //res[i * size + j] = 0; ----->> TO USE LOOP INTERCHAGE WE HAVE ASSUME THE RESULT MATRIX INIT WITH ZERO!
            for (int j = 0; j < size; j++) {
                res[i * size + j] += mat1[i * size + k] * mat2[k * size + j];
            }
        }
    }
}

void print_matrix(double* M, int size){
    printf("> Print Matrix:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.1f ", M[i * size + j]);
        }
        printf("\n");
    }

    printf(">inline: ");
    printf("{");
    for (int i = 0; i < size; i++) {
        printf("{");
        for (int j = 0; j < size; j++) {
            printf("%.1f ", M[i * size + j]);
            if( j != size - 1 ) printf(", ");
        }
        printf("}");
        if( i != size - 1 ) printf(", ");
    }
    printf("}\n\n");
}

int main(int argc, char* argv[]) {

    printf("Testing matrix ");

    int size = SIZE;
    if (argc == 2) {
        size = atoi( argv[1] );
    }

    printf("size: %d x %d\n", size, size);
    
    // Allocate memory for matrices
    double *A = (double*)malloc(size * size * sizeof(double));
    double *B = (double*)malloc(size * size * sizeof(double));
    double *C = (double*)malloc(size * size * sizeof(double));

    // Init matrix values pseudorandom

    // set initial seed for rand, 1 if debug-mode
    srand( DEBUG_FLAG ? 1 : time(NULL) );

    for(int i = 0; i < size * size; i++ ){
       A[i] = rand() % 10 + 1;
       B[i] = rand() % 10 + 1;
       C[i] = 0.0; // INIT ZERO FOR LOOP INTERCHAGE
    }

    IFDEBUG{
        print_matrix(A, size);
        print_matrix(B, size);
    }

    // Start timer
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply(A, B, C, size);

    // Stop timer
    clock_t end_time = clock();

    IFDEBUG{
        print_matrix(C, size);
    }

    // Calculate and print execution time
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : smatmul_loopinterchange, %f, %d\n", execution_time, size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
