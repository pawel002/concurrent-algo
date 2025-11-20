#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef uint64_t ll;
typedef uint8_t bitset_t;

#define BIT_INDEX(i) ((size_t)((i) >> 3))      // divides by 8
#define BIT_MASK(i) (uint8_t)(1u << ((i) & 7)) // 1 on i-th bit modulo 8

static inline bitset_t* bitset_calloc(size_t n_bits) 
{
    size_t n_bytes = (n_bits + 7) / 8;
    return (bitset_t*) calloc(n_bytes, 1);
}

static inline void bitset_set(bitset_t* b, size_t i) 
{
    b[BIT_INDEX(i)] |= BIT_MASK(i);
}

static inline int bitset_test(const bitset_t* b, size_t i) 
{
    return (b[BIT_INDEX(i)] & BIT_MASK(i)) != 0;
}

void run_sieve(bitset_t* is_prime, ll upper_bound)
{
    // mark numbers 1 and 0 as not prime
    bitset_set(is_prime, 0);
    bitset_set(is_prime, 1);

    // run sieve
    for (ll p = 2; p * p <= upper_bound; p++) 
    {
        if (!bitset_test(is_prime, p))
        { 
            for (ll x = p * p; x <= upper_bound; x += p) 
            {
                bitset_set(is_prime, x);
            }
        }
    }
}

ll count_primes(const bitset_t* is_prime, ll upper_bound) 
{
    ll base_count = 0;
    for (ll i = 0; i <= upper_bound; i++) 
        if (!bitset_test(is_prime, i)) 
            base_count++;
    return base_count;
}

void set_primes(const bitset_t* is_prime, ll upper_bound, ll* base_primes, ll offset)
{
    ll k = 0;
    for(ll i = 0; i <= upper_bound; i++)
        if(!bitset_test(is_prime, i))
            base_primes[k++] = i + offset;
}

void block_decompose(int rank, int size, ll upper_bound, ll N, ll* low, ll* high) 
{
    ll lowC = upper_bound + 1;
    ll lenC = N - lowC + 1;

    ll q = lenC / (ll) size; // base size of block
    ll r = lenC % (ll) size; // remainder

    // given first r block extra item
    ll add = (ll) (rank < (int) r ? rank : r);
    ll offset = (ll) rank * q + add;
    ll len = q + (rank < (int) r ? 1 : 0);

    // assign offsets
    *low = lowC + offset;
    *high = *low + len - 1;
}

void mark_base_primes(bitset_t* is_prime, ll* base_primes, ll base_prime_count, ll low, ll high)
{
    for(ll i = 0; i < base_prime_count; i++)
    {
        ll p = base_primes[i];
        ll start = (low + p - 1) / p * p;
        if (start < p * p) 
            start = p * p;

        for (ll x = start; x <= high; x += p)
            bitset_set(is_prime, (size_t)(x - low));
    }
}

int* gather_local_counts(int local_count, int rank, int size)
{
    int *counts = NULL;
    if (rank == 0)
        counts = (int*) malloc(size * sizeof(int));

    MPI_Gather(
        &local_count, 
        1, 
        MPI_INT,
        counts,        
        1, 
        MPI_INT,
        0, 
        MPI_COMM_WORLD
    );
    return counts;
}

int* gather_total_c_count(int* counts, int* total_primes, int rank, int size)
{   
    int* displs = NULL;
    if (rank != 0)
        return displs;

    displs = (int*) malloc(size * sizeof(int));
    displs[0] = 0;
    for (int i = 0; i < size; ++i) 
    {
        *total_primes += counts[i];
        if (i > 0) 
            displs[i] = displs[i-1] + counts[i-1];
    }
    return displs;
}

ll* gather_primes(ll* local_primes, ll local_prime_count, int* counts, 
    int* displs, int total_c_count, int rank)
{
    ll* all_c_primes = NULL;
    if (rank == 0)
        all_c_primes = (ll*) malloc((size_t)total_c_count * sizeof(ll));

    MPI_Gatherv(
        local_primes,            // pointer to memory we send
        local_prime_count,       // size of memory to be send
        MPI_UNSIGNED_LONG_LONG,  // sending type
        all_c_primes,            // pointer to receiving buffer (only in root)
        counts,                  // counts[i] tells how many elements root will get from rank i
        displs,                  // displs[i] starting index where rank i should start writing its data to recvbuf
        MPI_UNSIGNED_LONG_LONG,  // reciving type
        0,                       // rank number of (root)
        MPI_COMM_WORLD           // communicator
    );

    return all_c_primes;
}

void print_array(const ll* arr, ll size)
{
    for(ll i = 0; i<size; i++)
        printf("%llu, ", arr[i]);
}

// ========== SHARED SIEVE HELPERS ==========

void init_shared_comm(MPI_Comm* shared_comm, int* shared_size, int* shared_rank)
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

void calculate_shared_interval(ll* lowC, ll* lenC, size_t* n_bytes, ll N, ll upper_bound) 
{
    *lowC  = upper_bound + 1;
    *lenC  = N - *lowC + 1;
    *n_bytes = (size_t)((*lenC + 7) / 8);
}

void allocate_shared_bitset(MPI_Win* win, MPI_Comm* shared_comm, int shared_rank, size_t n_bytes)
{
    bitset_t *mybase = NULL;
    MPI_Aint winsize = (shared_rank == 0) ? (MPI_Aint) n_bytes : 0;
    int disp_unit = (int) sizeof(bitset_t);
    MPI_Win_allocate_shared(
        winsize, 
        disp_unit, 
        MPI_INFO_NULL, 
        *shared_comm,
        &mybase, 
        win
    );
}

bitset_t* get_shared_memory_pointer(MPI_Win* win, MPI_Comm* shared_comm, int shared_rank, size_t n_bytes)
{
    MPI_Aint query_size;
    int query_disp;
    void* baseptr = NULL;
    MPI_Win_shared_query(*win, 0, &query_size, &query_disp, &baseptr);
    bitset_t *is_prime_shm = (bitset_t*) baseptr;

    if (shared_rank == 0)
        memset(is_prime_shm, 0, n_bytes);

    MPI_Win_sync(*win);
    MPI_Barrier(*shared_comm);
    return baseptr;
}
