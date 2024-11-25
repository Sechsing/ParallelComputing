#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

void read_matrix(const char* filename, int** matrix, int* rows, int* cols) {
    FILE *file = fopen(filename, "rb");
    fread(rows, sizeof(int), 1, file);
    fread(cols, sizeof(int), 1, file);
    *matrix = (int*)malloc((*rows) * (*cols) * sizeof(int));  // Dereference matrix pointer correctly
    fread(*matrix, sizeof(int), (*rows) * (*cols), file);
    fclose(file);
}

void write_matrix(const char* filename, unsigned long long* matrix, int rows, int cols) {
    FILE *file = fopen(filename, "wb");
    fwrite(&rows, sizeof(int), 1, file);
    fwrite(&cols, sizeof(int), 1, file);
    fwrite(matrix, sizeof(unsigned long long), rows * cols, file);
    fclose(file);
}

int main(int argc, char** argv) {
    // Set the number of threads
    omp_set_num_threads(2); // Explicitly set to 2 threads
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    double bcast_time, scatter_time, gather_time;

    int rowA, colA, rowB, colB;  // Declare matrix dimensions
    int* pMatrixA = NULL;  // Declare matrix A pointer
    int* pMatrixB = NULL;  // Declare matrix B pointer

    if (rank == 0) {
        start_time = MPI_Wtime(); // Start timing

        // Use OpenMP to read the input matrices
        printf("Reading Matrix A and B - Start\n");

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                read_matrix("MA_1000x1000.bin", &pMatrixA, &rowA, &colA);
            }
            #pragma omp section
            {
                read_matrix("MB_1000x1000.bin", &pMatrixB, &rowB, &colB);
            }
        }

        printf("Reading Matrix A and B - Done\n");
    }

    // Broadcast the matrix dimensions
    MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD); // Added broadcast for rowA
    MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD); // Added broadcast for colA

    // Allocate memory for matrix B
    if (rank != 0) {
        pMatrixA = (int*)malloc(rowA * colA * sizeof(int)); // Allocate for other processes
        pMatrixB = (int*)malloc(rowB * colB * sizeof(int));
    }

    // Broadcast matrix B to all processes
    double start_bcast = MPI_Wtime();
    MPI_Bcast(pMatrixB, rowB * colB, MPI_INT, 0, MPI_COMM_WORLD);
    double end_bcast = MPI_Wtime();
    bcast_time = end_bcast - start_bcast;

    // Calculate local workload: a tile of matrix A
    int tileSize = 500; // Tile size for partitioning that is to be manually changed to be number of matrices/2
    int numTiles = (rowA / tileSize);
    int numTilesPerProcess = numTiles / size;
    int extraTiles = numTiles % size;

    int localRows = tileSize * numTilesPerProcess;
    if (rank < extraTiles) {
        localRows += tileSize;
    }

    int* localA = (int*)malloc(localRows * colA * sizeof(int));
    unsigned long long* localC = (unsigned long long*)calloc(localRows * colB, sizeof(unsigned long long));

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

    // Perform local matrix multiplication using OpenMP threads
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < colB; j++) {
            for (int k = 0; k < colA; k++) {
                localC[i * colB + j] += localA[i * colA + k] * pMatrixB[k * colB + j];
            }
        }
    }

    // Gather the local results back to the root process
    double start_gather = MPI_Wtime();
    if (rank == 0) {
        unsigned long long* pMatrixC = (unsigned long long*)calloc(rowA * colB, sizeof(unsigned long long));
        MPI_Gatherv(localC, localRows * colB, MPI_UNSIGNED_LONG_LONG, pMatrixC, sendcounts, displs, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        // Use OpenMP to write the matrix result
        printf("Writing Matrix C to File - Start\n");
        write_matrix("MC_mpi.bin", pMatrixC, rowA, colB);
        printf("Writing Matrix C to File - Done\n");

        end_time = MPI_Wtime(); // End timing
        printf("Total execution time: %f seconds\n", end_time - start_time); // Output time
        printf("Broadcast time: %f seconds\n", bcast_time);
        printf("Scatter time: %f seconds\n", scatter_time);
        printf("Gather time: %f seconds\n", gather_time);

        free(pMatrixC);
    } else {
        MPI_Gatherv(localC, localRows * colB, MPI_UNSIGNED_LONG_LONG, NULL, NULL, NULL, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    free(localA);
    free(localC);
    if (rank == 0) {
        free(pMatrixA);
        free(pMatrixB);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}