#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define SHIFT_X 0
#define SHIFT_Y 1
#define SHIFT_Z 2
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
    int ndims = 3, size, my_rank, reorder, my_cart_rank, ierr;
    int nrows, ncols, ndepth;
    int nbr_x_lo, nbr_x_hi, nbr_y_lo, nbr_y_hi, nbr_z_lo, nbr_z_hi;
    MPI_Comm comm3D;
    int dims[ndims], coord[ndims];
    int wrap_around[ndims];
    MPI_Request requests[6];  // For non-blocking sends and receives

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    srand(time(NULL) + my_rank);  // Seed random number generator with rank for uniqueness

    // Process command line arguments (nrows, ncols, ndepth)
    if (argc == 4) {
        nrows = atoi(argv[1]);
        ncols = atoi(argv[2]);
        ndepth = atoi(argv[3]);
        dims[0] = nrows;
        dims[1] = ncols;
        dims[2] = ndepth;
        if ((nrows * ncols * ndepth) != size) {
            if (my_rank == 0)
                printf("ERROR: nrows*ncols*ndepth=%d * %d * %d = %d != %d\n", nrows, ncols, ndepth, nrows * ncols * ndepth, size);
            MPI_Finalize();
            return 0;
        }
    } else {
        nrows = ncols = ndepth = (int)cbrt(size);
        dims[0] = dims[1] = dims[2] = 0;
    }

    // Create Cartesian topology
    MPI_Dims_create(size, ndims, dims);
    wrap_around[0] = wrap_around[1] = wrap_around[2] = 0;  // No periodic boundary conditions
    reorder = 1;

    ierr = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, wrap_around, reorder, &comm3D);
    if (ierr != 0) {
        printf("ERROR[%d] creating CART\n", ierr);
        MPI_Finalize();
        return 0;
    }

    // Find coordinates and rank in Cartesian communicator
    MPI_Cart_coords(comm3D, my_rank, ndims, coord);
    MPI_Cart_rank(comm3D, coord, &my_cart_rank);

    // Get neighbors (left, right, top, bottom, front, back)
    MPI_Cart_shift(comm3D, SHIFT_X, DISP, &nbr_x_lo, &nbr_x_hi);
    MPI_Cart_shift(comm3D, SHIFT_Y, DISP, &nbr_y_lo, &nbr_y_hi);
    MPI_Cart_shift(comm3D, SHIFT_Z, DISP, &nbr_z_lo, &nbr_z_hi);

    // Print rank, coordinates, and neighbors
    printf("Rank %d: Cartesian coords = (%d, %d, %d). Neighbors -> Left: %d, Right: %d, Top: %d, Bottom: %d, Front: %d, Rear: %d\n",
           my_rank, coord[0], coord[1], coord[2], nbr_x_lo, nbr_x_hi, nbr_y_lo, nbr_y_hi, nbr_z_lo, nbr_z_hi);

    // Open a log file for this process with buffering
    char filename[50];
    sprintf(filename, "rank_%d.txt", my_rank);
    FILE *log_file = fopen(filename, "w");
    setvbuf(log_file, NULL, _IOFBF, 1024);  // Set buffer size for the log file

    // Loop for NUM_ITERATIONS
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        // Generate a random prime number
        int my_prime = generate_random_prime();
        int left_prime = -1, right_prime = -1, top_prime = -1, bottom_prime = -1, front_prime = -1, rear_prime = -1;

        // Non-blocking communication with neighbors in 6 directions
        MPI_Isend(&my_prime, 1, MPI_INT, nbr_x_lo, 0, comm3D, &requests[0]);
        MPI_Irecv(&right_prime, 1, MPI_INT, nbr_x_hi, 0, comm3D, &requests[1]);

        MPI_Isend(&my_prime, 1, MPI_INT, nbr_x_hi, 1, comm3D, &requests[2]);
        MPI_Irecv(&left_prime, 1, MPI_INT, nbr_x_lo, 1, comm3D, &requests[3]);

        MPI_Isend(&my_prime, 1, MPI_INT, nbr_y_lo, 2, comm3D, &requests[4]);
        MPI_Irecv(&bottom_prime, 1, MPI_INT, nbr_y_hi, 2, comm3D, &requests[5]);

        MPI_Isend(&my_prime, 1, MPI_INT, nbr_y_hi, 3, comm3D, &requests[6]);
        MPI_Irecv(&top_prime, 1, MPI_INT, nbr_y_lo, 3, comm3D, &requests[7]);

        MPI_Isend(&my_prime, 1, MPI_INT, nbr_z_lo, 4, comm3D, &requests[8]);
        MPI_Irecv(&rear_prime, 1, MPI_INT, nbr_z_hi, 4, comm3D, &requests[9]);

        MPI_Isend(&my_prime, 1, MPI_INT, nbr_z_hi, 5, comm3D, &requests[10]);
        MPI_Irecv(&front_prime, 1, MPI_INT, nbr_z_lo, 5, comm3D, &requests[11]);

        // Wait for all communications to complete
        MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

        // Compare primes and log matches
        if (my_prime == left_prime && left_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_x_lo, my_rank);
        }
        if (my_prime == right_prime && right_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_x_hi, my_rank);
        }
        if (my_prime == top_prime && top_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_y_lo, my_rank);
        }
        if (my_prime == bottom_prime && bottom_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_y_hi, my_rank);
        }
        if (my_prime == front_prime && front_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_z_lo, my_rank);
        }
        if (my_prime == rear_prime && rear_prime != -1) {
            fprintf(log_file, "Match: %d from rank %d to rank %d\n", my_prime, nbr_z_hi, my_rank);
        }

        fflush(log_file);  // Ensure logging is up to date after each iteration
    }

    // Close the log file
    fclose(log_file);

    // Finalize MPI
    MPI_Comm_free(&comm3D);
    MPI_Finalize();
    
    return 0;
}

