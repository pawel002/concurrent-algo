import subprocess
import time
import matplotlib.pyplot as plt

def compile_code():
    print("Compiling...")
    subprocess.run(["nvcc", "distribution/main.cu", "-o", "main", "-lcurand", "-O3"])
    print("Compilation successful.")

def run_benchmark(n, method):
    times = []
    cmd = [EXECUTABLE, str(n), method, "0"]

    for _ in range(RUNS_PER_DATAPOINT):

        t = time.perf_counter()
        subprocess.run(cmd, capture_output=True, text=True)
        t = time.perf_counter() - t
        
        times.append(t)
    
    avg_time = sum(times) / len(times)
    return avg_time


EXECUTABLE = "./main"
N_VALUES = [1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]
METHODS = ["naive", "batch"]
RUNS_PER_DATAPOINT = 5

compile_code()

results = {method: [] for method in METHODS}
    
print(f"{'N':<15} {'Method':<10} {'Avg Time (s)':<15}")
print("-" * 40)

for n in N_VALUES:
    for method in METHODS:
        avg_time = run_benchmark(n, method)
        results[method].append(avg_time)
        print(f"{n:<15} {method:<10} {avg_time:.6f}")

plt.figure(figsize=(10, 6))

for method in METHODS:
    plt.plot(N_VALUES, results[method], marker='o', label=method)

plt.xscale('log')
plt.xlabel('N (Number of particles)')
plt.ylabel('Time (seconds)')
plt.title('CUDA Random Walk Performance: Naive vs Batch')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

output_file = 'data/benchmark_plot.png'
plt.savefig(output_file)