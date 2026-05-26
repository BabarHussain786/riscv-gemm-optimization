#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define IFDEBUG if(DEBUG_FLAG)

#define DEBUG_FLAG 0
#define SIZE 64

//in theory the size of tile must fill as much as possibile the L1-cache
//so, in case of matmul, the size of tail was
//
// size_tail = sqrt[ (L1Cache in byte) / sizeof(double) ] 
//
// so for example on a 32KB, we have 32768 Byte cache size, the best will be 64
// because in cache can be 4096 double, squared in a 64 x 64
//
// But we must try, depends also by the architecure... some today as unbalanced!
// for example on my x86 decacore i7-1255U with L1d=32KB, the there is a good increse 
// up to 32 but the size 128 is better (a few better)
//
// ERROR: => we need 3 tiles in cache??? (not only one! ..)
//
// BANANA_PI => L1-d := 32K
// ----> total = 3 x (s x s) x sizeof(double) <= 32K
//    -----> s = 36  ===>  s = 32
// BUT TRY: {8, 16, 32, 64, 128}
#define DEFAULT_TILE_SIZE 64 


// Function to perform matrix multiplication
// ts: tile size, passed as parameter (for tuning)
void multiply(double* mat1, double* mat2, double* res, int size, int ts) {


      #pragma omp parallel for
      for( int ih = 0; ih < size; ih += ts ){      //parallel
      
          #pragma omp paralle for
          for( int jh = 0; jh < size; jh += ts ){  //parallel

              for( int kh = 0; kh < size; kh += ts){
                  for( int il = 0; il < ts; ++il ){
                      for( int kl = 0; kl < ts; ++kl ){
                          float ax_prefetch = mat1[(ih + il) * size + (kh + kl)];
                          for( int jl = 0; jl < ts; ++jl ){
                              res[ (ih + il) * size  + (jh + jl) ] += 
                                  ax_prefetch * 
                                  mat2[(kh + kl) * size + (jh + jl)];
                          }
                      }
                  }
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
    
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    printf("OMP NESTED ON!\n");

    int size = SIZE;
    int tile_size = DEFAULT_TILE_SIZE;

    if (argc == 2) {
        size = atoi( argv[1] );
    }
    else if (argc == 3) {
        size = atoi( argv[1] );
        tile_size = atoi( argv[2] );
    }

    printf("size: %d x %d    tile_size:%d\n", size, size, tile_size);
    printf("Num proc: %d\n", omp_get_max_threads());

    // check tile_size is correct ( tile_size <= size )
    if( tile_size > size ){
        printf("ERROR: tile_size:%d must be less than or equal to size:%d\n", tile_size, size);
        exit(EXIT_FAILURE);
    }
    
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
       C[i] = 0.0;
    }

    IFDEBUG{
        print_matrix(A, size);
        print_matrix(B, size);
    }

    // Start timer
    double start_time = omp_get_wtime();

    // Perform matrix multiplication (GEMM)
    multiply(A, B, C, size, tile_size);

    // Stop timer
    double end_time = omp_get_wtime();

    IFDEBUG{
        print_matrix(C, size);
    }

    // Calculate and print execution time
    double execution_time = (end_time - start_time); // [sec]

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : smatmul_tiling_v4, %f, %d, %d\n", execution_time, size, tile_size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
