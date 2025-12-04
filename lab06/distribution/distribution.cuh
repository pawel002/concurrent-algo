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
void naive_kernel(
    float* __restrict__ x,
    curandState* __restrict__ globalState,
    const int steps,
    const float dt_sqrt,
    const int n
)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx >= n) return;

    curandState localState = globalState[tx];
    float val = 0.0f;

    for (int i = 0; i < steps; i++) {
        float noise = curand_normal(&localState);
        val = val + dt_sqrt * noise;
    }

    x[tx] = val;
}

float* generate_naive(
    const int n,
    const float tau,
    const float dt
)
{
    const float dt_sqrt = sqrtf(dt);
    float *d_x = nullptr;
    curandState *d_states = nullptr;

    size_t bytes = (size_t) n * sizeof(float);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_states, n * sizeof(curandState));

    int blocks = (n + config::threads - 1) / config::threads;
    init_rng_kernel<<<blocks, config::threads>>>(d_states, time(NULL), n);
    cudaDeviceSynchronize();

    const int steps = (int)(tau / dt);
    naive_kernel<<<blocks, config::threads>>>(
        d_x, d_states, steps, dt_sqrt, n
    );

    float *h_x = (float*) malloc(bytes);
    cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_states);

    return h_x;
}

__global__
void sum_rows_kernel(
    const float* __restrict__ d_normal,
    float* __restrict__ x,
    int n,
    int steps,
    float dt_sqrt
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float acc = 0.0f;
    int base = row * steps;

    for (int j = 0; j < steps; ++j)
        acc += d_normal[base + j];

    x[row] = acc * dt_sqrt;
}

float* generate_batchrand(
    const int n,
    const float tau,
    const float dt
)
{
    const int steps = (int)(tau / dt);
    const float dt_sqrt = sqrtf(dt);
    const size_t total = (size_t)(n * steps);
    const size_t bytes_normal = total * sizeof(float);
    const size_t bytes_vec    = (size_t)(n * sizeof(float));

    float *d_normal = nullptr, *d_x = nullptr, *h_x = nullptr;

    cudaMalloc(&d_normal, bytes_normal);
    cudaMalloc(&d_x, bytes_vec);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long) time(NULL));
    curandGenerateNormal(gen, d_normal, total, 0.0f, 1.0f);

    int blocks = (n + config::threads - 1) / config::threads;
    sum_rows_kernel<<<blocks, config::threads>>>(
        d_normal, d_x, n, steps, dt_sqrt
    );
    cudaDeviceSynchronize();

    h_x = (float*) malloc(bytes_vec);
    cudaMemcpy(h_x, d_x, bytes_vec, cudaMemcpyDeviceToHost);

    curandDestroyGenerator(gen);
    cudaFree(d_normal);
    cudaFree(d_x);

    return h_x;
}