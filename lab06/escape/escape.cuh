#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

namespace config {
    constexpr int threads = 1024;
}

__global__ 
void init_rng_kernel(
    curandState *state, 
    unsigned long seed, 
    int n
)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    
    curand_init(seed, id, 0, &state[id]);
}

__global__ 
void simulation_kernel(
    int* __restrict__ t,
    curandState* __restrict__ globalState,
    const float dt_sqrt,
    const int n,
    const int T
)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx >= n) return;

    curandState localState = globalState[tx];
    float val = 0.0f;

    for (int i = 0; i < T; i++) {
        float noise = curand_normal(&localState);
        val = val + dt_sqrt * noise;

        if (val < -1.0f || val > 1.0f) {
            t[tx] = i + 1;
            return;
        }
    }
}

int* simulate(
    const int n,
    const float dt,
    const int T
)
{
    const float dt_sqrt = sqrtf(dt);
    int *d_x = nullptr;
    curandState *d_states = nullptr;

    size_t bytes = (size_t) n * sizeof(int);
    cudaMalloc(&d_x, bytes);
    cudaMemset(d_x, 0, bytes); 

    cudaMalloc(&d_states, n * sizeof(curandState));

    int blocks = (n + config::threads - 1) / config::threads;
    init_rng_kernel<<<blocks, config::threads>>>(d_states, time(NULL), n);
    cudaDeviceSynchronize();

    simulation_kernel<<<blocks, config::threads>>>(
        d_x, d_states, dt_sqrt, n, T
    );
    cudaDeviceSynchronize();

    int *h_x = (int*) malloc(bytes);
    cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_states);
    cudaFree(d_x); 

    return h_x;
}