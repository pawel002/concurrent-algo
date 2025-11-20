#include "helpers.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // initialize communicator for the world
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // initialized communicator per node
    MPI_Comm shared_comm;
    int shared_rank, shared_size;
    init_shared_comm(&shared_comm, &shared_size, &shared_rank);

    // define prime scannig interval
    const ll N = 1000000000;
    const ll lower_bound = 2;
    const ll upper_bound = (ll) floor(sqrt(N));

    // make is_prime lookup, is_prime[k] = 0 => k is prime
    bitset_t* is_prime = bitset_calloc(upper_bound + 1);
    run_sieve(is_prime, upper_bound);

    // collect primes into array
    ll base_prime_count = count_primes(is_prime, upper_bound);
    ll* base_primes = (ll*) malloc(base_prime_count * sizeof(ll));
    set_primes(is_prime, upper_bound, base_primes, 0);
    free(is_prime);

    // calculate parameters of interval C = [upper_bound+1, N]
    ll lowC, lenC;
    size_t n_bytes;
    calculate_shared_interval(&lowC, &lenC, &n_bytes, N, upper_bound);

    // allocate shared memory window
    MPI_Win win;
    allocate_shared_bitset(&win, &shared_comm, shared_rank, n_bytes);

    // get the pointer to shared memory in each process
    bitset_t* is_prime_shared = get_shared_memory_pointer(&win, &shared_comm, shared_rank, n_bytes);

    // decompose domain C = [upper_bound+1, N]
    ll low, high;
    block_decompose(world_rank, world_size, upper_bound, N, &low, &high);

    // local sieve
    ll local_len = high - low + 1;
    bitset_t *is_prime_seg = is_prime_shared + BIT_INDEX((size_t)(low - lowC));
    mark_base_primes(is_prime_seg, base_primes, base_prime_count, low, high);

    // publish local writes
    MPI_Win_sync(win);
    MPI_Barrier(shared_comm);

    int base_count_int = (int) base_prime_count;
    int single_node = (shared_size == world_size);

    if (single_node) {
        // Everyone is on one node -> rank 0 can walk the entire shared bitset
        ll c_prime_count = count_primes(is_prime_shared, lenC - 1);
        int total_primes = base_count_int + (int) c_prime_count;

        ll *all_c_primes = (ll*) malloc((size_t)c_prime_count * sizeof(ll));
        set_primes(is_prime_shared, lenC - 1, all_c_primes, lowC);

        if (world_rank == 0) {
            printf("n = %llu\n", N);
            printf("Primes in B = [2..%llu]: %d\n", upper_bound, base_count_int);
            printf("Primes in C = [%llu..%llu]: %d\n", lowC, N, (int)c_prime_count);
            printf("Total primes in [2..%llu]: %d\n", N, total_primes);
        }
        free(all_c_primes);
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shared_comm);
    MPI_Finalize();
    return 0;
}