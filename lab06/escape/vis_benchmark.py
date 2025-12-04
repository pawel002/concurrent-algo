import subprocess
import time
import matplotlib.pyplot as plt

def compile_code():
    print("Compiling...")
    subprocess.run(["nvcc", "escape/main.cu", "-o", "main", "-lcurand", "-O3"])
    print("Compilation successful.")

def run_benchmark(n):
    times = []
    cmd = [EXECUTABLE, str(n), "0"]

    for _ in range(RUNS_PER_DATAPOINT):

        t = time.perf_counter()
        subprocess.run(cmd, capture_output=True, text=True)
        t = time.perf_counter() - t
        
        times.append(t)
    
    avg_time = sum(times) / len(times)
    return avg_time


EXECUTABLE = "./main"
N_VALUES = [1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]
RUNS_PER_DATAPOINT = 5

compile_code()

results = []
    
print(f"{'N':<15} {'Avg Time (s)':<15}")
print("-" * 40)

for n in N_VALUES:
    avg_time = run_benchmark(n)
    results.append(avg_time)
    print(f"{n:<15} {avg_time:.6f}")

plt.figure(figsize=(10, 6))

plt.plot(N_VALUES, results, marker='o', label="Escape sim time")

plt.xscale('log')
plt.xlabel('N (Number of particles)')
plt.ylabel('Time t [s]')
plt.title('CUDA Escape simulation')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

output_file = 'data/benchmark_escape_plot.png'
plt.savefig(output_file)