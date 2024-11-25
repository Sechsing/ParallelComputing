#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int my_rank;
    int p;
    int val = -1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    do {
        if (my_rank == 0) {
            printf("Enter a positive integer (> 0): ");
            fflush(stdout);
            scanf("%d", &val);
        }

        // Missing line: Broadcast the value to all processes
        MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);

        printf("Processor %d received value: %d\n", my_rank, val);
        fflush(stdout);

    } while (val > 0);

    MPI_Finalize();
    return 0;
}

