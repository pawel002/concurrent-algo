import matplotlib.pyplot as plt
import numpy as np

TAU = 2.5 

data = np.fromfile("data/results.bin", dtype=np.float32)

counts, bin_edges = np.histogram(data, bins=200, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(10, 6))

plt.fill_between(bin_centers, counts, alpha=0.6, color='#007acc', label='Generated Data')
plt.step(bin_centers, counts, where='mid', color='#005a9e', linewidth=1)

x = np.linspace(data.min(), data.max(), 100)
p = (1 / np.sqrt(2 * np.pi * TAU)) * np.exp(-x**2 / (2 * TAU))

plt.plot(x, p, 'r', linewidth=2, label=r"$N(0, \tau)$")

plt.title(f"Distribution of Generated Random Numbers (N={len(data)})")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

output_file = "data/distribution_plot.png"
plt.savefig(output_file)