#include "executor.h"

#define PI 3.14159265358979323846

int main(int argc, char** argv) 
{
    const float radius = 100.0f;
    const int N = 12;
    const int s = 0; // set to 1 to save graph x, y and solution into file

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int initial_depth = DEFAULT_INITIAL_DEPTH;
    if (argc > 1) {
        initial_depth = atoi(argv[1]);
        if (initial_depth < 1) initial_depth = 1;
    }

    MPI_Datatype task_type;
    create_task_type(&task_type);
    
    MPI_Datatype result_type;
    create_result_type(&result_type); 

    Graph g; 
    g.n = N;

    for(int i = 0; i < g.n; i++)
    {
        float angle = (2.0 * PI * i) / (float) N;
        g.x[i] = radius * cos(angle);
        g.y[i] = radius * sin(angle);        
    }

    srand(42);
    for (int i = g.n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        float temp_x = g.x[i]; g.x[i] = g.x[j]; g.x[j] = temp_x;
        float temp_y = g.y[i]; g.y[i] = g.y[j]; g.y[j] = temp_y;
    }

    for(int i=0; i<g.n; i++) 
        for(int j=0; j<g.n; j++)
            g.dist[i][j] = calc_dist(g.x[i], g.y[i], g.x[j], g.y[j]);

    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time = MPI_Wtime();

    if (rank == 0) master(size - 1, &g, task_type, result_type, s, initial_depth);
    else worker(rank, &g, task_type, result_type);

    MPI_Barrier(MPI_COMM_WORLD); 
    double end_time = MPI_Wtime();

    if (rank == 0)
        printf("Execution Time: %f seconds\n", end_time - start_time);

    MPI_Type_free(&task_type);
    MPI_Type_free(&result_type);
    MPI_Finalize();
    return 0;
}