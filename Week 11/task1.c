#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 2

typedef struct {
    int thread_id;
    int row_start;
    int row_end;
    int colA;
    int colB;
    int *pMatrixA;
    int *pMatrixB;
    unsigned long long *pMatrixC;
} ThreadData;

void *matrix_multiply(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int i, j, k;
    
    for (i = data->row_start; i < data->row_end; i++) {
        for (j = 0; j < data->colB; j++) {
            for (k = 0; k < data->colA; k++) {
                data->pMatrixC[i * data->colB + j] += 
                    (data->pMatrixA[i * data->colA + k] * data->pMatrixB[k * data->colB + j]);
            }
        }
    }
    pthread_exit(NULL);
}

int main() {
    int i, rowA, colA, rowB, colB;
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Timing variables
    struct timespec start, end;
    double time_taken;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 1. Read Matrix A
    printf("Reading Matrix A - Start\n");
    FILE *pFileA = fopen("MA_4000x4000.bin", "rb");
    fread(&rowA, sizeof(int), 1, pFileA);
    fread(&colA, sizeof(int), 1, pFileA);
    int *pMatrixA = (int*)malloc((rowA * colA) * sizeof(int));
    for(i = 0; i < rowA; i++) {
        fread(&pMatrixA[i * colA], sizeof(int), colA, pFileA);
    }
    fclose(pFileA);
    printf("Reading Matrix A - Done\n");

    // 2. Read Matrix B
    printf("Reading Matrix B - Start\n");
    FILE *pFileB = fopen("MB_4000x4000.bin", "rb");
    fread(&rowB, sizeof(int), 1, pFileB);
    fread(&colB, sizeof(int), 1, pFileB);
    int *pMatrixB = (int*)malloc((rowB * colB) * sizeof(int));
    for(i = 0; i < rowB; i++) {
        fread(&pMatrixB[i * colB], sizeof(int), colB, pFileB);
    }
    fclose(pFileB);
    printf("Reading Matrix B - Done\n");

    // 3. Initialize result matrix C
    unsigned long long *pMatrixC = (unsigned long long*)calloc((rowA * colB), sizeof(unsigned long long));

    // 4. Create Threads
    int rows_per_thread = rowA / NUM_THREADS;
    for (i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].row_start = i * rows_per_thread;
        thread_data[i].row_end = (i == NUM_THREADS - 1) ? rowA : (i + 1) * rows_per_thread;
        thread_data[i].colA = colA;
        thread_data[i].colB = colB;
        thread_data[i].pMatrixA = pMatrixA;
        thread_data[i].pMatrixB = pMatrixB;
        thread_data[i].pMatrixC = pMatrixC;
        pthread_create(&threads[i], NULL, matrix_multiply, (void *)&thread_data[i]);
    }

    // 5. Join Threads
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // 6. Write results to file
    printf("Writing Resultant Matrix C to File - Start\n");
    FILE *pFileC = fopen("MC.bin", "wb");
    fwrite(&rowA, sizeof(int), 1, pFileC);
    fwrite(&colB, sizeof(int), 1, pFileC);
    for (i = 0; i < rowA; i++) {
        fwrite(&pMatrixC[i * colB], sizeof(unsigned long long), colB, pFileC);
    }
    fclose(pFileC);
    printf("Writing Resultant Matrix C to File - Done\n");

    // Timing end
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
    printf("Overall time (s): %lf\n", time_taken);

    // Clean up
    free(pMatrixA);
    free(pMatrixB);
    free(pMatrixC);

    return 0;
}

