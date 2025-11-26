#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#define MAX_NODES 20
#define INITIAL_DEPTH 1

#define TAG_TASK 1
#define TAG_RESULT 2
#define TAG_KILL 3

typedef struct {
    int n;
    float x[MAX_NODES];
    float y[MAX_NODES];
    float dist[MAX_NODES][MAX_NODES]; 
} Graph;

typedef struct {
    int path[MAX_NODES]; 
    int count;           
    float current_cost;  
    float lower_bound;
} Task;

typedef struct {
    float cost;
    int path[MAX_NODES];
} SearchResult;

// helpers
float calc_dist(float x1, float y1, float x2, float y2) 
{
    return sqrtf(powf(x1 - x2, 2) + powf(y1 - y2, 2));
}

float compute_bound(Graph *g, Task *t) 
{
    if (t->count == g->n) return t->current_cost;
    return t->current_cost;
}

void save_coords(const char* filename, Graph* g) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error opening coordinates file");
        return;
    }

    fprintf(f, "ID\tX\tY\n");
    for (int i = 0; i < g->n; i++) {
        fprintf(f, "%d\t%.4f\t%.4f\n", i, g->x[i], g->y[i]);
    }

    fclose(f);
    printf("Coordinates saved to %s\n", filename);
}

void save_solution(const char* filename, int* path, int n, float cost) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error opening solution file");
        return;
    }

    fprintf(f, "Optimal Cost: %.4f\n", cost);

    for (int i = 0; i < n; i++) {
        fprintf(f, "%d ", path[i]);
    }

    fprintf(f, "%d\n", path[0]);

    fclose(f);
    printf("Solution saved to %s\n", filename);
}

void create_task_type(MPI_Datatype *dt) 
{
    int blocks[4] = {MAX_NODES, 1, 1, 1};
    MPI_Aint disps[4];
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
    Task dummy;

    MPI_Get_address(&dummy.path, &disps[0]);
    MPI_Get_address(&dummy.count, &disps[1]);
    MPI_Get_address(&dummy.current_cost, &disps[2]);
    MPI_Get_address(&dummy.lower_bound, &disps[3]);

    for(int i=3; i>=0; i--) 
        disps[i] -= disps[0];

    MPI_Type_create_struct(4, blocks, disps, types, dt);
    MPI_Type_commit(dt);
}

void create_result_type(MPI_Datatype *dt) {
    int blocks[2] = {1, MAX_NODES};
    MPI_Aint disps[2];
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_INT};

    SearchResult dummy;
    MPI_Get_address(&dummy.cost, &disps[0]);
    MPI_Get_address(&dummy.path, &disps[1]);
    disps[1] -= disps[0];
    disps[0] = 0;

    MPI_Type_create_struct(2, blocks, disps, types, dt);
    MPI_Type_commit(dt);
}

// solve for subtree
void solve_subtree_recursive(Graph *g, Task t, float *local_best_cost, int *local_best_path) {
    if (t.lower_bound >= *local_best_cost) return;

    if (t.count == g->n) 
    {
        float total = t.current_cost + g->dist[t.path[t.count-1]][t.path[0]];
        if (total < *local_best_cost) 
        {
            *local_best_cost = total;
            memcpy(local_best_path, t.path, sizeof(int) * g->n);
        }
        return;
    }

    int last_node = t.path[t.count - 1];
    
    for (int i = 0; i < g->n; i++) 
    {
        int visited = 0;
        for(int k=0; k<t.count; k++) if(t.path[k] == i) visited = 1;
        
        if (!visited) 
        {
            Task next = t;
            next.path[next.count] = i;
            next.count++;
            next.current_cost += g->dist[last_node][i];
            next.lower_bound = compute_bound(g, &next);

            if (next.lower_bound < *local_best_cost)
                solve_subtree_recursive(g, next, local_best_cost, local_best_path);
        }
    }
}

// worker node
void worker(int rank, Graph *g, MPI_Datatype task_type, MPI_Datatype result_type) 
{
    Task t;
    MPI_Status status;
    SearchResult res;

    while(1) 
    {
        MPI_Recv(&t, 1, task_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_KILL) break;

        res.cost = FLT_MAX;
        solve_subtree_recursive(g, t, &res.cost, res.path);

        MPI_Send(&res, 1, result_type, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

// master node
void master(int num_workers, Graph *g, MPI_Datatype task_type, MPI_Datatype result_type, int save) 
{
    Task *queue = (Task*) malloc(sizeof(Task) * 50000); 
    int q_head = 0;
    int q_tail = 0;
    
    float global_best_cost = FLT_MAX;
    int global_best_path[MAX_NODES];

    Task start_task = { .count = 1, .current_cost = 0, .lower_bound = 0 };
    start_task.path[0] = 0;
    queue[q_tail++] = start_task;

    printf("Master: Generating prefixes up to depth %d...\n", INITIAL_DEPTH);

    while (q_head != q_tail) 
    {
        if (queue[q_head].count >= INITIAL_DEPTH)
            break; 
        
        Task t = queue[q_head++];
        int last_node = t.path[t.count - 1];

        for (int i = 0; i < g->n; i++) 
        {
            int visited = 0;
            for(int k = 0; k < t.count; k++) 
                if(t.path[k] == i) visited = 1;
            
            if (!visited) 
            {
                Task next = t;
                next.path[next.count] = i;
                next.count++;
                next.current_cost += g->dist[last_node][i];
                next.lower_bound = compute_bound(g, &next);
                
                queue[q_tail++] = next;
            }
        }
    }

    int total_tasks = q_tail - q_head;
    printf("Master: Generated %d sub-problems.\n", total_tasks);

    // dispatch tasks
    int tasks_sent = 0;
    int tasks_completed = 0;
    int active_workers = 0;

    // initial call
    for (int w = 1; w <= num_workers; w++) 
    {
        if (q_head < q_tail) 
        {
            MPI_Send(&queue[q_head++], 1, task_type, w, TAG_TASK, MPI_COMM_WORLD);
            tasks_sent++;
            active_workers++;
        }
    }

    // dynamic loop
    while (tasks_completed < total_tasks) 
    {
        SearchResult res;
        MPI_Status status;
        
        MPI_Recv(&res, 1, result_type, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
        
        active_workers--;
        tasks_completed++;
        
        if (res.cost < global_best_cost) {
            global_best_cost = res.cost;
            // COPY PATH locally
            memcpy(global_best_path, res.path, sizeof(int) * g->n);
            printf("New Best: %f\n", global_best_cost);
        }

        if (q_head < q_tail) 
        {
            MPI_Send(&queue[q_head++], 1, task_type, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
            active_workers++;
        }
    }

    printf("Master: All tasks done. Best Cost: %f\n", global_best_cost);

    // cleanup
    for (int w = 1; w <= num_workers; w++) 
    {
        MPI_Send(NULL, 0, MPI_INT, w, TAG_KILL, MPI_COMM_WORLD);
    }
    free(queue);

    // save to file
    if (save)
    {
        save_coords("data/coords.txt", g);
        save_solution("data/solution.txt", global_best_path, g->n, global_best_cost);
    }
}
