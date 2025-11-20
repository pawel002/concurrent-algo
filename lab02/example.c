#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = -1, size = -1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 5) {
        if (rank == 0) {
            fprintf(stderr, "Program wymaga dokładnie 5 procesów (size=%d)\n", size);
        }
        MPI_Abort(comm, 1);
    }

    const int N = 25;
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "N (%d) must be divisible by number of processes (%d)\n", N, size);
        }
        MPI_Abort(comm, 2);
    }
    const int chunk = N / size;

    // Rank 0 allocates, others expose 0 bytes in the same shared window
    const int disp_unit = (int)sizeof(int32_t);
    MPI_Aint nbytes = (rank == 0) ? (MPI_Aint)N * disp_unit : 0;

    void* baseptr = NULL;
    MPI_Win win;
    MPI_Win_allocate_shared(nbytes, disp_unit, MPI_INFO_NULL, comm, &baseptr, &win);

    // Get pointer to rank-0's shared segment
    void* baseptr0 = NULL;
    MPI_Aint sz = 0;
    int disp_out = 0;
    MPI_Win_shared_query(win, 0, &sz, &disp_out, &baseptr0);

    int32_t* shared = (int32_t*)baseptr0;

    // Initialize (rank 0)
    if (rank == 0) {
        for (int i = 0; i < N; ++i) shared[i] = -1;
    }

    MPI_Barrier(comm);   // make sure initialization is done

    // Each rank writes its id into its chunk
    const int start = rank * chunk;
    const int end   = start + chunk;
    for (int i = start; i < end; ++i) {
        shared[i] = (int32_t)rank;
    }

    // For shared-memory load/store: flush private copies, then sync all ranks
    MPI_Win_sync(win);
    MPI_Barrier(comm);

    if (rank == 0) {
        printf("Zawartość współdzielonej tablicy:\n[");
        for (int i = 0; i < N; ++i) {
            printf("%d%s", shared[i], (i + 1 < N) ? ", " : "");
        }
        printf("]\n");
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
