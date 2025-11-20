from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import time
from collections import defaultdict

x0, x1, w = -2.0, +2.0, 640 * 2
y0, y1, h = -1.5, +1.5, 480 * 2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)

ENABLE_SLEEP = True
SLEEP_LINE = 1 
SLEEP_TIME = 2.0


def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + c
        n -= 1
    return n


def julia_line(k):
    rank = MPI.COMM_WORLD.Get_rank()

    t_start = time.perf_counter()

    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)

    t_line = time.perf_counter() - t_start

    if ENABLE_SLEEP and k == SLEEP_LINE:
        print(f"[rank {rank}] linia {k}: usypiam na {SLEEP_TIME:.3f} s")
        time.sleep(SLEEP_TIME)
        t_line += SLEEP_TIME

    print(f"[rank {rank}] linia {k}: czas = {t_line:.6f} s")
    return k, line, rank, t_line


if __name__ == '__main__':
    t_program_start = time.perf_counter()

    with MPIPoolExecutor() as executor:
        results = list(executor.map(julia_line, range(h)))

    t_program_end = time.perf_counter()
    total_program_time = t_program_end - t_program_start

    results_sorted = sorted(results, key=lambda r: r[0])

    with open('julia.pgm', 'wb') as f:
        header = f'P5 {w} {h} 255\n'.encode('ascii')
        f.write(header)
        for k, line_bytes, rank_line, t_line in results_sorted:
            f.write(line_bytes)

    line_times = [t_line for (k, line_bytes, rank_line, t_line) in results]

    # pkt 2: min, max, różnica
    min_time = min(line_times)
    max_time = max(line_times)
    diff_time = max_time - min_time
    print("\n== Statystyki czasów linii (pkt 2) ==")
    print(f"min: {min_time:.6f} s")
    print(f"max: {max_time:.6f} s")
    print(f"różnica (max - min): {diff_time:.6f} s")

    # pkt 3: sprawdzenie przypisania linii do procesów
    print("\n== Przypisanie linii do procesów (pkt 3) ==")
    print("Pierwsze 50 linii:")
    for k, line_bytes, rank_line, t_line in sorted(results, key=lambda r: r[0])[:50]:
        print(f"linia {k:4d} -> rank {rank_line}")

    # pkt 4 i 5: liczba linii na proces oraz suma czasów na proces
    lines_per_rank = defaultdict(int)
    time_per_rank = defaultdict(float)

    for k, line_bytes, rank_line, t_line in results:
        lines_per_rank[rank_line] += 1
        time_per_rank[rank_line] += t_line

    print("\n== Liczba linii na każdy proces (pkt 4) ==")
    for rnk in sorted(lines_per_rank):
        print(f"rank {rnk}: {lines_per_rank[rnk]} linii")

    print("\n== Suma czasów linii na każdy proces (pkt 5) ==")
    for rnk in sorted(time_per_rank):
        print(f"rank {rnk}: {time_per_rank[rnk]:.6f} s")

    print("\n== Czas całego programu (pkt 5) ==")
    print(f"całkowity czas programu (od startu MPIPoolExecutor do końca obliczeń): "
          f"{total_program_time:.6f} s")

    print("\nSuma czasów wszystkich linii (dla porównania): "
          f"{sum(line_times):.6f} s")

