#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    long N;
    double local_sum = 0.0, total_sum = 0.0;
    double piVal;
    struct timespec start, end;
    double time_taken;

    MPI_Init(&argc, &argv);  // Initialize the MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);  // Get the number of processes

    // The root rank (rank 0) prompts the user for the value of N
    if (my_rank == 0) {
        printf("Enter the number of intervals (N): ");
        fflush(stdout);
        scanf("%ld", &N);
    }

    // Broadcast the value of N to all processes
    MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Get the start time after broadcasting N
    if (my_rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    // Each process calculates its part of the sum
    long local_n = N / comm_sz;  // Number of intervals per process
    long start_idx = my_rank * local_n;
    long end_idx = (my_rank + 1) * local_n;

    for (long i = start_idx; i < end_idx; i++) {
        local_sum += 4.0 / (1 + pow((2.0 * i + 1.0) / (2.0 * N), 2));
    }

    // Reduce all local sums to a total sum in the root process
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // The root process calculates the final value of Pi and prints the results
    if (my_rank == 0) {
        piVal = total_sum / (double)N;

        // Get the end time
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Calculate the time taken in seconds
        time_taken = (end.tv_sec - start.tv_sec) * 1e9;
        time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;

        printf("Calculated Pi value (Parallel MPI) = %12.9f\n", piVal);
        printf("Overall time (s): %lf\n", time_taken);
    }

    MPI_Finalize();  // Finalize the MPI environment
    return 0;
}

