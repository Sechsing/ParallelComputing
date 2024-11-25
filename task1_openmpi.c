#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int get_user_input() {
    int n;
    printf("Enter a number: ");
    scanf("%d", &n);
    return n;
}

int main(int argc, char *argv[]) {
    int rank, size, n;
    int *local_primes, *all_primes = NULL;
    int local_count = 0, total_count = 0;
    int offset = 0;
    double start_time, end_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process gets the input from the user
        n = get_user_input();
    }

    // Broadcast the value of n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Start timing after broadcast
    start_time = MPI_Wtime();

    // Calculate the range of numbers for each process
    int range = (n - 2 + size - 1) / size; // Divide the numbers evenly among processes
    int start = rank * range + 2;
    int end = (start + range - 1 < n) ? start + range - 1 : n - 1;

    // Find primes in the range
    local_primes = (int *)malloc((end - start + 1) * sizeof(int));
    for (int i = start; i <= end; i++) {
        if (is_prime(i)) {
            local_primes[local_count++] = i;
        }
    }

    if (rank == 0) {
        // Root process collects data from all processes
        all_primes = (int *)malloc(n * sizeof(int));  // Overestimate size for all primes

        // Copy root's local primes to all_primes array
        for (int i = 0; i < local_count; i++) {
            all_primes[offset++] = local_primes[i];
        }

        // Receive prime data from other processes
        for (int p = 1; p < size; p++) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(all_primes + offset, count, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += count;
        }

        // Stop timing after collecting all data
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;

        // Print all primes
        printf("Prime numbers less than %d are:\n", n);
        for (int i = 0; i < offset; i++) {
            printf("%d ", all_primes[i]);
        }
        printf("\n");

        printf("Elapsed time: %f seconds\n", elapsed_time);

        free(all_primes);
    } else {
        // Non-root processes send their data to the root process
        MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_primes, local_count, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Stop timing for non-root processes
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
    }

    free(local_primes);
    MPI_Finalize();
    return 0;
}

