import os
import time
import subprocess
import matplotlib.pyplot as plt

def time_once(cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0

def build_cmd(process_count, prefix_count):
    return ["mpiexec", "-n", str(process_count), "./salesman", str(prefix_count)]

def bench_grid(grid_size, thread_counts, repeats):
    means = []

    for p in thread_counts:
        runs = [time_once(build_cmd(p, grid_size)) for _ in range(repeats)]
        mean = sum(runs) / repeats
        means.append(mean)
        print(f"N={grid_size}, p={p}: runs={', '.join(f'{r:.3f}s' for r in runs)} | mean={mean:.3f}s")

    return means

def compute_metrics(runtimes_by_grid, thread_counts):
    speedup_by_grid = {}
    efficiency_by_grid = {}
    for N, means in runtimes_by_grid.items():

        T1 = means[0]
        s = [T1 / m for m in means]           
        e = [si / p for si, p in zip(s, thread_counts)]
        
        speedup_by_grid[N] = s
        efficiency_by_grid[N] = e
    return speedup_by_grid, efficiency_by_grid

def plot_lines(y_by_grid, thread_counts, title, y_label, out_file, karp=False, repeats=1):
    plt.figure(figsize=(8, 5))

    for N, ys in sorted(y_by_grid.items()):
        xs = thread_counts[1:] if karp else thread_counts
        plt.plot(xs, ys if not karp else ys, marker="o", label=f"N={N}")

    plt.xlabel("Thread count (processes)")
    plt.ylabel(f"{y_label} (avg of {repeats} runs)")
    plt.title(title)
    plt.legend(title="Grid size")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()

def main():
    PREFIX_COUNTS = [2, 3, 4]
    THREAD_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    REPEATS       = 3

    # 1) collect runtimes
    runtimes_by_grid = {N: bench_grid(N, THREAD_COUNTS, REPEATS) for N in PREFIX_COUNTS}

    # 2) plot runtime (one line per grid size)
    plot_lines(
        y_by_grid=runtimes_by_grid,
        thread_counts=THREAD_COUNTS,
        title="Runtime vs thread count",
        y_label="Wall time [s]",
        out_file="data/benchmarks/runtime_vs_threads.png",
        karp=False,
        repeats=REPEATS
    )

    # 3) compute derived metrics
    speedup_by_grid, efficiency_by_grid, karp_by_grid = compute_metrics(runtimes_by_grid, THREAD_COUNTS)

    # 4) plot speedup
    plot_lines(
        y_by_grid=speedup_by_grid,
        thread_counts=THREAD_COUNTS,
        title="Speedup vs thread count",
        y_label="Speedup (T1/Tp)",
        out_file="data/benchmarks/speedup_vs_threads.png",
        karp=False,
        repeats=REPEATS
    )

    # 5) plot efficiency
    plot_lines(
        y_by_grid=efficiency_by_grid,
        thread_counts=THREAD_COUNTS,
        title="Efficiency vs thread count",
        y_label="Efficiency (Speedup / p)",
        out_file="data/benchmarks/efficiency_vs_threads.png",
        karp=False,
        repeats=REPEATS
    )

    # 6) plot Karp–Flatt (note: p>1 only)
    plot_lines(
        y_by_grid=karp_by_grid,
        thread_counts=THREAD_COUNTS,
        title="Karp-Flatt metric vs thread count",
        y_label="Karp-Flatt ε",
        out_file="data/benchmarks/karp_flatt_vs_threads.png",
        karp=True,
        repeats=REPEATS
    )

if __name__ == "__main__":
    main()
