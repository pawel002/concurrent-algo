import matplotlib.pyplot as plt
import numpy as np

data = np.fromfile("data/results_escape.bin", dtype=np.int32)

escaped = data[data > 0]
not_escaped = len(data) - len(escaped)

print(f"Total:     {len(data)}")
print(f"Escaped:   {len(escaped)}")
print(f"Trapped:   {not_escaped}")

DT = 1e-4
T = 50000
T_final = DT * T
t_samples = escaped * DT

def survival_tau(t, n_terms=200):
    t = np.atleast_1d(t).astype(np.float64)

    m = np.arange(n_terms, dtype=np.float64)
    coeff = (-1.0)**m / (2.0 * m + 1.0)
    lam = ((2.0 * m + 1.0)**2) * (np.pi**2) / 8.0 

    exponents = np.exp(-lam[:, None] * t[None, :])
    S = (4.0 / np.pi) * np.dot(coeff, exponents)

    return S if S.size > 1 else S[0]


def pdf_tau(t, n_terms=200):
    t = np.atleast_1d(t).astype(np.float64)

    m = np.arange(n_terms, dtype=np.float64)
    a = (-1.0)**m / (2.0 * m + 1.0)
    lam = ((2.0 * m + 1.0)**2) * (np.pi**2) / 8.0

    exponents = np.exp(-lam[:, None] * t[None, :])
    f = (4.0 / np.pi) * np.dot(a * lam, exponents)

    return f if f.size > 1 else f[0]


p_escape_theory = 1.0 - survival_tau(T_final)
t_grid = np.linspace(0.0, T_final, 2000)[1:]
f_grid = pdf_tau(t_grid)

f_cond = f_grid / p_escape_theory

plt.figure(figsize=(10, 6))

plt.hist(
    t_samples,
    bins=200,
    density=True,
    color='#ff7f0e',
    edgecolor='black',
    alpha=0.6,
    label='Simulation'
)

plt.plot(
    t_grid,
    f_cond,
    'k',
    linewidth=2.0,
    label='Theoretical (first exit pdf)'
)

plt.title("Distribution of First Passage Times")
plt.xlabel("Time t")
plt.ylabel("Density")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("data/escape_plot.png", dpi=150)
