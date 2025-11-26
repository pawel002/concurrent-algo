import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def time_once(cmd):
    t0 = time.perf_counter()
    print("benchmarking ", cmd)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0

def build_cmd(process_count, prefix_count):
    return ["mpiexec", "-n", str(process_count), "./salesman", str(prefix_count)]

def bench_grid(prefix_count, thread_counts, repeats):
    means = []
    stds = []
    for p in thread_counts:
        runs = [time_once(build_cmd(p, prefix_count)) for _ in range(repeats)]
        means.append(np.mean(runs))
        stds.append(np.std(runs))
    return np.array(means), np.array(stds)

def compute_metrics(runtimes, thread_counts):
    results = {}
    for N, (means, stds) in runtimes.items():
        t1 = means[0]
        speedup = t1 / means
        efficiency = speedup / thread_counts
        
        sf = []
        sf_threads = []
        for s, p in zip(speedup, thread_counts):
            if p > 1:
                e = ((1/s) - (1/p)) / (1 - (1/p))
                sf.append(e)
                sf_threads.append(p)
        
        results[N] = {
            "mean": means,
            "std": stds,
            "speedup": speedup,
            "efficiency": efficiency,
            "serial_fraction": np.array(sf),
            "sf_threads": np.array(sf_threads)
        }

    return results

def plot_all(results, thread_counts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = [
        ("Runtime vs Processes", "mean", "Time [s]", "runtime.png", True),
        ("Speedup vs Processes", "speedup", "Speedup (T1/Tp)", "speedup.png", False),
        ("Efficiency vs Processes", "efficiency", "Efficiency (S/p)", "efficiency.png", False),
        ("Serial Fraction (Karp-Flatt)", "serial_fraction", "Serial Fraction", "serial_fraction.png", False)
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    
    for title, key, ylabel, fname, use_band in metrics:
        plt.figure(figsize=(8, 5))
        
        for (N, data), color in zip(sorted(results.items()), colors):
            x = thread_counts
            y = data[key]
            
            if key == "serial_fraction":
                x = data["sf_threads"]
            
            plt.plot(x, y, marker="o", label=f"Depth={N}", color=color)
            
            if use_band:
                std = data["std"]
                plt.fill_between(x, y - std, y + std, color=color, alpha=0.2)
                
        plt.xlabel("Process count")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

def main():
    PREFIX_COUNTS = [2, 3, 4]
    THREAD_COUNTS = np.arange(1, 11)
    REPEATS = 10
    
    runtimes = {N: bench_grid(N, THREAD_COUNTS, REPEATS) for N in PREFIX_COUNTS}
    metrics = compute_metrics(runtimes, THREAD_COUNTS)
    plot_all(metrics, THREAD_COUNTS, "data/benchmarks")

if __name__ == "__main__":
    main()