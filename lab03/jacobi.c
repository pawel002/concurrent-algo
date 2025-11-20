#include "helpers.h"

#define x(i, j) X[(i)*(N) + (j)]
#define y(i, j) X_new[(i)*(N) + (j)]

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int N            = -1;             // grid size
    const int T      = 20000;          // max iteration count
    const int report = 1000;           // check norm every
    const double h   = 1.0/(double)N;  // grid spacing (normalised to grid 1x1)
    const double g   = 1.0;            // g constant
    const double lam = 1.0;            // lambda
    const double C   = h*h * (g/lam);  // collapsed constant for Jacobi update
    const double eps = 1e-5;           // L2 tolerance on difference between iterates
    const int save   = 0;              // 1 - save solution to file, 0 - don't save

    // initialize communicator for the world
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank, world_size;
    init_mpi(comm, &world_rank, &world_size);

    // parse and broadcast N
    parse_and_brodcast(&N, world_rank, argv, comm);

    // check if world_size % 2 == 0
    if (!verify_args(world_rank, world_size))
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // initialized communicator per node
    MPI_Comm shared_comm;
    int shared_rank, shared_size;
    init_shared_comm(&shared_comm, &shared_size, &shared_rank);

    // allocate two one continous memory space of size 2 * N^2 for arrays X and X_new
    MPI_Win win;
    const MPI_Aint bytes_total = (size_t)(2 * N * N * sizeof(double));
    allocate_shared_memory(&win, &shared_comm, shared_rank, bytes_total);

    // get the pointer to shared memory in each process and assign to X and X_new with offset
    double* base = get_shared_memory_pointer(&win, &shared_comm, shared_rank, bytes_total);
    double* X = base;
    double* X_new = base + N * N;  // N^2 offset

    // blocks are defined as (start_i, start_j) - (end_i, end_j), find them for each process
    int start_i, start_j, end_i, end_j;
    block_decompose(world_rank, world_size, N, &start_i, &start_j, &end_i, &end_j);

    // main loop
    double *tmp;
    int stop = 0;
    for(int t=0; t<T; t++)
    {   
        // update values
        for(int i=start_i; i<end_i; i++)
        {
            for(int j=start_j; j<end_j; j++)
            {
                y(i, j) = 0.25 * (x(i + 1, j) + x(i - 1, j) + x(i, j + 1) + x(i, j - 1) + C); 
            }
        }
        
        // sync
        MPI_Win_sync(win);
        MPI_Barrier(shared_comm);
        
        // every 10 iteration check L2 norm between X and X_new, only on root
        if (t % report == 0)
        {
            double local = local_residual(X, X_new, N, start_i, start_j, end_i, end_j);
            double global = 0.0;
            MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
            global = sqrt(global);

            if (world_rank == 0) {
                fprintf(stderr, "[%6d] residual = %.6e\n", t, global);
            }
            stop = (global < eps);
        }
        
        // send the value of stop to each thread
        MPI_Bcast(&stop, 1, MPI_INT, 0, comm);
        if (stop) break;

        // swap pointers
        tmp = X;
        X = X_new; 
        X_new = tmp;

        // ensure swap
        MPI_Barrier(shared_comm);
    }

    if (save == 1 && world_rank == 0) {
        write_to_file("data/grids/grid_1024.bin", X, N);
        printf("Done");
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shared_comm);
    MPI_Finalize();
    return 0;
}