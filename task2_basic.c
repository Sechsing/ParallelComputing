#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1
#define NUM_ITERATIONS 500
#define MAX_RANDOM 1000  // Upper bound for random number generation

// Function to check if a number is prime
int is_prime(int num) {
    if (num <= 1) return 0;
    if (num == 2 || num == 3) return 1;
    if (num % 2 == 0 || num % 3 == 0) return 0;
    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return 0;
    }
    return 1;
}

// Function to generate a random prime number
int generate_random_prime() {
    int num;
    do {
        num = rand() % MAX_RANDOM;  // Generate a random number between 0 and MAX_RANDOM
    } while (!is_prime(num));  // Keep generating until a prime number is found
    return num;
}

int main(int argc, char *argv[]) {
    int ndims = 2, size, my_rank, reorder, my_cart_rank, ierr;
    int nrows, ncols;
    int nbr_i_lo, nbr_i_hi;
    int nbr_j_lo, nbr_j_hi;
    MPI_Comm comm2D;
    int dims[ndims], coord[ndims];
    int wrap_around[ndims];

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    srand(time(NULL) + my_rank);  // Seed random number generator with rank for uniqueness

    // Process command line arguments
    if (argc == 3) {
        nrows = atoi(argv[1]);
        ncols = atoi(argv[2]);
        dims[0] = nrows;
        dims[1] = ncols;
        if ((nrows * ncols) != size) {
            if (my_rank == 0)
                printf("ERROR: nrows*ncols=%d * %d = %d != %d\n", nrows, ncols, nrows * ncols, size);
            MPI_Finalize();
            return 0;
        }
    } else {
        nrows = ncols = (int)sqrt(size);
        dims[0] = dims[1] = 0;
    }

    // Create Cartesian topology
    MPI_Dims_create(size, ndims, dims);
    wrap_around[0] = wrap_around[1] = 0;  // No periodic boundary conditions
    reorder = 1;

    ierr = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, wrap_around, reorder, &comm2D);
    if (ierr != 0) {
        printf("ERROR[%d] creating CART\n", ierr);
        MPI_Finalize();
        return 0;
    }

    // Find coordinates and rank in Cartesian communicator
    MPI_Cart_coords(comm2D, my_rank, ndims, coord);
    MPI_Cart_rank(comm2D, coord, &my_cart_rank);

    // Get neighbors (left, right, top, bottom)
    MPI_Cart_shift(comm2D, SHIFT_ROW, DISP, &nbr_i_lo, &nbr_i_hi);
    MPI_Cart_shift(comm2D, SHIFT_COL, DISP, &nbr_j_lo, &nbr_j_hi);

    // Open a log file for this process
    char filename[50];
    sprintf(filename, "rank_%d.txt", my_rank);
    FILE *log_file = fopen(filename, "w");

    // Loop for NUM_ITERATIONS
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        // Generate a random prime number
        int my_prime = generate_random_prime();
        int left_prime = -1, right_prime = -1, top_prime = -1, bottom_prime = -1;

        // Exchange prime numbers with neighbors
        MPI_Sendrecv(&my_prime, 1, MPI_INT, nbr_j_lo, 0, &right_prime, 1, MPI_INT, nbr_j_hi, 0, comm2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&my_prime, 1, MPI_INT, nbr_j_hi, 1, &left_prime, 1, MPI_INT, nbr_j_lo, 1, comm2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&my_prime, 1, MPI_INT, nbr_i_lo, 2, &bottom_prime, 1, MPI_INT, nbr_i_hi, 2, comm2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&my_prime, 1, MPI_INT, nbr_i_hi, 3, &top_prime, 1, MPI_INT, nbr_i_lo, 3, comm2D, MPI_STATUS_IGNORE);

        // Compare primes and log matches
        if (my_prime == left_prime && left_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_j_lo, my_rank);
        }
        if (my_prime == right_prime && right_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_j_hi, my_rank);
        }
        if (my_prime == top_prime && top_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_i_lo, my_rank);
        }
        if (my_prime == bottom_prime && bottom_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_i_hi, my_rank);
        }

        fflush(log_file);  // Ensure logging is up to date after each iteration
    }

    // Close the log file
    fclose(log_file);

    // Finalize MPI
    MPI_Comm_free(&comm2D);
    MPI_Finalize();

    return 0;
}
