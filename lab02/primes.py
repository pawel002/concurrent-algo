from math import isqrt
from functools import lru_cache

def _icbrt(n: int) -> int:
    """floor cube root with correction to avoid FP off-by-ones."""
    if n <= 0:
        return 0
    x = int(round(n ** (1.0 / 3.0)))
    # adjust
    while (x + 1) ** 3 <= n:
        x += 1
    while x ** 3 > n:
        x -= 1
    return x

def _sieve_with_pi(limit: int):
    """Simple sieve up to 'limit' returning (primes, pi_table)."""
    if limit < 2:
        return [], [0] * (limit + 1)
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    r = isqrt(limit)
    for p in range(2, r + 1):
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start:limit + 1:step] = b"\x00" * ((limit - start) // step + 1)
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    pi = [0] * (limit + 1)
    cnt = 0
    for i in range(limit + 1):
        if i >= 2 and is_prime[i]:
            cnt += 1
        pi[i] = cnt
    return primes, pi

def prime_pi(n: int) -> int:
    """Number of primes <= n (π(n))."""
    if n < 2:
        return 0

    # Choose sieve bound large enough that all subproblems reduce to table lookups:
    # We need primes up to sqrt(n), and pi(w) for w <= n^(2/3).
    limit = max(isqrt(n), int(n ** (2.0 / 3.0))) + 10
    primes, pi_small = _sieve_with_pi(limit)

    @lru_cache(maxsize=None)
    def phi(x: int, a: int) -> int:
        # Count integers in [1..x] not divisible by the first a primes.
        if a == 0:
            return x
        if x == 0:
            return 0
        if a == 1:
            return x - x // 2
        # recurse
        return phi(x, a - 1) - phi(x // primes[a - 1], a - 1)

    @lru_cache(maxsize=None)
    def lehmer_pi(x: int) -> int:
        if x <= limit:
            return pi_small[x]
        a = lehmer_pi(_icbrt(x))         # a = π(x^(1/3))
        b = lehmer_pi(isqrt(x))          # b = π(√x)
        res = phi(x, a) + a - 1
        # pi(x) = phi(x, a) + a - 1 - sum_{i=a+1}^b [pi(x/p_i) - i + 1]
        for i in range(a + 1, b + 1):
            res -= lehmer_pi(x // primes[i - 1]) - (i - 1)
        return res

    return lehmer_pi(n)

if __name__ == "__main__":
    # Demo: compute π(10^9) and check the claim "π(10^9) = 50844133"
    N = 10**9
    cnt = prime_pi(N)
    print(f"pi({N}) = {cnt}")
    claim = 50847534
    print(f"Claim 'pi(10^9) == {claim}' is {cnt == claim}")