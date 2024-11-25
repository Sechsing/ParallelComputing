#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    double bcast_time, scatter_time, gather_time;

    if (rank == 0) {
        printf("Matrix Multiplication using MPI - Start\n");
    }

    int rowA = 500, colA = 500, rowB = 500, colB = 500;
    int tileSize = 250; 

    int* pMatrixA = NULL;
    int* pMatrixB = NULL;
    unsigned long long* pMatrixC = NULL;
    unsigned long long* localC = NULL;

    // Root process reads matrix A and B
    if (rank == 0) {
        printf("Reading Matrix A - Start\n");
        
        // Reading matrix A
        FILE *pFileA = fopen("MA_500x500.bin", "rb");
        fread(&rowA, sizeof(int), 1, pFileA);
        fread(&colA, sizeof(int), 1, pFileA);
        pMatrixA = (int*)malloc(rowA * colA * sizeof(int));
        for (int i = 0; i < rowA; i++) {
            fread(&pMatrixA[i * colA], sizeof(int), colA, pFileA);
        }
        fclose(pFileA);
        printf("Reading Matrix A - Done\n");

        printf("Reading Matrix B - Start\n");
        
        // Reading matrix B
        FILE *pFileB = fopen("MB_500x500.bin", "rb");
        fread(&rowB, sizeof(int), 1, pFileB);
        fread(&colB, sizeof(int), 1, pFileB);
        pMatrixB = (int*)malloc(rowB * colB * sizeof(int));
        for (int i = 0; i < rowB; i++) {
            fread(&pMatrixB[i * colB], sizeof(int), colB, pFileB);
        }
        fclose(pFileB);
        printf("Reading Matrix B - Done\n");

        // Initialize matrix C
        pMatrixC = (unsigned long long*)calloc(rowA * colB, sizeof(unsigned long long));
    }

    // Broadcast matrix B to all processes
    if (rank != 0) {
        pMatrixB = (int*)malloc(rowB * colB * sizeof(int));
    }

    double start_bcast = MPI_Wtime();
    MPI_Bcast(pMatrixB, rowB * colB, MPI_INT, 0, MPI_COMM_WORLD);
    double end_bcast = MPI_Wtime();
    bcast_time = end_bcast - start_bcast;

    // Calculate local workload: a tile of matrix A
    int numTiles = (rowA / tileSize);
    int numTilesPerProcess = numTiles / size;
    int extraTiles = numTiles % size;

    int localRows = tileSize * numTilesPerProcess;
    if (rank < extraTiles) {
        localRows += tileSize;
    }

    int* localA = (int*)malloc(localRows * colA * sizeof(int));
    localC = (unsigned long long*)calloc(localRows * colB, sizeof(unsigned long long));

    // Scatter tiles of matrix A to each process
    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = tileSize * numTilesPerProcess * colA;
            if (i < extraTiles) {
                sendcounts[i] += tileSize * colA;
            }
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    double start_scatter = MPI_Wtime();
    MPI_Scatterv(pMatrixA, sendcounts, displs, MPI_INT, localA, localRows * colA, MPI_INT, 0, MPI_COMM_WORLD);
    double end_scatter = MPI_Wtime();
    scatter_time = end_scatter - start_scatter;

    if (rank == 0) {
        printf("Matrix Multiplication - Start\n");
    }

    // Perform local matrix multiplication
    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < colB; j++) {
            for (int k = 0; k < colA; k++) {
                localC[i * colB + j] += localA[i * colA + k] * pMatrixB[k * colB + j];
            }
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication - Done\n");
    }

    // Gather the local results back to the root process
    double start_gather = MPI_Wtime();
    if (rank == 0) {
        MPI_Gatherv(localC, localRows * colB, MPI_UNSIGNED_LONG_LONG, pMatrixC, sendcounts, displs, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(localC, localRows * colB, MPI_UNSIGNED_LONG_LONG, NULL, NULL, NULL, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }
    double end_gather = MPI_Wtime();
    gather_time = end_gather - start_gather;

    if (rank == 0) {
        printf("MPI communication time (s) for rank %d: %f\n", rank, scatter_time + bcast_time);
        printf("Write Resultant Matrix C to File - Start\n");

        // Root process writes the final result to a binary file
        FILE *pFileC = fopen("MC_mpi.bin", "wb");
        fwrite(&rowA, sizeof(int), 1, pFileC);
        fwrite(&colB, sizeof(int), 1, pFileC);
        for (int i = 0; i < rowA; i++) {
            fwrite(&pMatrixC[i * colB], sizeof(unsigned long long), colB, pFileC);
        }
        fclose(pFileC);

        printf("Write Resultant Matrix C to File - Done\n");

        end_time = MPI_Wtime();
        printf("Overall time (Including read, multiplication and write)(s): %f\n", end_time - start_time); // Output time
        printf("MPI communication time (s): %f\n", bcast_time + scatter_time + gather_time);
        printf("Matrix Multiplication using MPI - Done\n");
    }

    // Cleanup
    free(localA);
    free(localC);
    if (rank == 0) {
        free(pMatrixA);
        free(pMatrixB);
        free(pMatrixC);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
