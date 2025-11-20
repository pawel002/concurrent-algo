#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline void init_mpi(MPI_Comm comm, int* rank, int* size)
{
    MPI_Comm_rank(comm, rank);
    MPI_Comm_size(comm, size);
}

static inline void parse_and_brodcast(int* N, int world_rank, char** argv, MPI_Comm comm) 
{
    if (world_rank == 0) 
    {
        char *end = NULL;
        *N = (int) strtol(argv[1], &end, 10);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, comm);
}

static inline int verify_args(int world_rank, int world_size) 
{
    if (world_size % 2 == 1 && world_size != 1)
    {
        if (world_rank == 0)
            fprintf(stderr, "Error: MPI world size (%d) must be a multiple of 2 or equal to 1.\n", world_size);
        return 0;
    }
    return 1;
}

static inline void init_shared_comm(MPI_Comm* shared_comm, int* shared_size, int* shared_rank)
{
    MPI_Comm_split_type(
        MPI_COMM_WORLD, 
        MPI_COMM_TYPE_SHARED, 
        0,
        MPI_INFO_NULL, 
        shared_comm
    );

    MPI_Comm_rank(*shared_comm, shared_rank);
    MPI_Comm_size(*shared_comm, shared_size);
}

static inline void allocate_shared_memory(MPI_Win* win, MPI_Comm* shared_comm, int shared_rank, size_t n_bytes)
{
    double* mybase = NULL;
    MPI_Aint winsize = (shared_rank == 0) ? (MPI_Aint) n_bytes : 0;
    int disp_unit = (int) sizeof(double);
    MPI_Win_allocate_shared(
        winsize, 
        disp_unit, 
        MPI_INFO_NULL, 
        *shared_comm,
        &mybase, 
        win
    );
}

static inline double* get_shared_memory_pointer(MPI_Win* win, MPI_Comm* shared_comm, int shared_rank, size_t n_bytes)
{
    MPI_Aint query_size;
    int query_disp;
    void* baseptr = NULL;
    MPI_Win_shared_query(*win, 0, &query_size, &query_disp, &baseptr);

    // zero the entire region
    if (shared_rank == 0)
        memset(baseptr, 0, n_bytes);

    MPI_Win_sync(*win);
    MPI_Barrier(*shared_comm);
    return (double *)baseptr;
}

// split [1..N-2] interior among p parts -> [start,end) in interior indices
static inline void split_1d_interior(int N, int p, int idx, int *start, int *end) {
    const int interior = N - 2;
    const int base = interior / p;
    const int rem  = interior % p;
    const int len  = base + (idx < rem ? 1 : 0);
    const int off  = idx * base + (idx < rem ? idx : rem);
    *start = 1 + off;
    *end   = *start + len;
}

// 2D block decomposition: 2 rows x (world_size/2) cols
static inline void block_decompose(int world_rank, int world_size, int N,
                                   int *si, int *sj, int *ei, int *ej) {
    // if single thread, then give it entire matrix to handle
    if(world_size == 1)
    {
        *si = 1;
        *sj = 1;
        *ei = N - 1;
        *ej = N - 1;
        return;
    }

    const int p_rows = 2;
    const int p_cols = world_size / 2;
    const int row = world_rank / p_cols;
    const int col = world_rank % p_cols;

    split_1d_interior(N, p_rows, row, si, ei);
    split_1d_interior(N, p_cols, col, sj, ej);
}

// local L2 sum over a block
static inline double local_residual(const double *X, const double *Y, int N, 
                                    int si, int sj, int ei, int ej) {
    double s = 0.0;
    for (int i = si; i < ei; ++i) {
        int io = i * N;
        for (int j = sj; j < ej; ++j) {
            double d = Y[io + j] - X[io + j];
            s += d * d;
        }
    }
    return s;
}

static inline int write_to_file(const char* fname, double* X, int N)
{
    FILE *f = fopen(fname, "wb");
    if (!f) {
        perror("fopen");
        return 0;
    }

    // write N
    fwrite(&N, sizeof(int), 1, f);

    // write matrix
    size_t nwrite = fwrite(X, sizeof(double), (size_t)(N * N), f);
    if (nwrite != (size_t)(N * N)) 
    {
        fprintf(stderr, "Short write.\n");
        return 0;
    }

    fclose(f);
    return 1;
}