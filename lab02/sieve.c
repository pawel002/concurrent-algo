#include "helpers.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = -1, size = -1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // define interval and make is_prime lookup
    const ll N = 1000000000;
    const ll lower_bound = 2;
    const ll upper_bound = (ll) floor(sqrt(N));
    bitset_t* is_prime = bitset_calloc(upper_bound + 1);

    run_sieve(is_prime, upper_bound);

    // collect primes into array
    ll base_prime_count = count_primes(is_prime, upper_bound);
    ll* base_primes = (ll*) malloc(base_prime_count * sizeof(ll));
    set_primes(is_prime, upper_bound, base_primes, 0);
    free(is_prime);

    // decompose domain C = [upper_bound+1, N]
    ll low = 0, high = 0;
    block_decompose(rank, size, upper_bound, N, &low, &high);

    // local sieve
    ll local_len = high - low + 1;
    bitset_t* is_prime_local = bitset_calloc(local_len);
    mark_base_primes(is_prime_local, base_primes, base_prime_count, low, high);

    // collect local primes
    ll local_prime_count = count_primes(is_prime_local, local_len - 1);
    int local_count_i = (int) local_prime_count;

    ll* local_primes = (ll*) malloc(local_prime_count * sizeof(ll));
    set_primes(is_prime_local, local_len - 1, local_primes, low);
    free(is_prime_local);

    // counts holds amount of primes computed in each thread
    int* counts = gather_local_counts(local_count_i, rank, size);

    // find total number of primes and memory displacement array
    int total_c_count = 0;
    int* displs = gather_total_c_count(counts, &total_c_count, rank, size);

    // gather the actual primes into root
    ll* all_c_primes = gather_primes(local_primes, local_prime_count, counts, displs, total_c_count, rank);

    int base_count_int = (int) base_prime_count;
    int total_primes = base_count_int + total_c_count;

    if (rank == 0) {
        printf("n = %llu\n", N);
        printf("Primes in B = [2..%llu]: %d\n", upper_bound, base_count_int);
        printf("Primes in C = [%llu..%llu]: %d\n", upper_bound + 1, N, total_c_count);
        printf("Total primes in [2..%llu]: %d\n", N, total_primes);
    }

    MPI_Finalize();
}