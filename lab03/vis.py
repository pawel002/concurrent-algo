import numpy as np
import matplotlib.pyplot as plt

def load_data(path: str):
    with open(path, "rb") as f:
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        X = np.fromfile(f, dtype=np.float64, count=N*N).reshape(N, N)
        return X
    

def visualise_surface_and_heatmap(Z):
    N = Z.shape[0]

    step = max(1, N // 200)
    Zi = Z[::step, ::step]
    j = np.arange(0, N, step)
    i = np.arange(0, N, step)
    Xg, Yg = np.meshgrid(j, i)

    fig = plt.figure(figsize=(10, 4))
    fig.canvas.manager.set_window_title("Surface + Heatmap")

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(Xg, Yg, Zi, linewidth=0, antialiased=True)
    ax3d.set_xlabel("j (column)")
    ax3d.set_ylabel("i (row)")
    ax3d.set_zlabel("T")
    ax3d.set_title(f"Solution surface (N={N})")

    ax2d = fig.add_subplot(1, 2, 2)
    im = ax2d.imshow(Z, origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax2d)
    cbar.set_label("T")
    ax2d.set_title(f"Solution heatmap (N={N})")
    ax2d.set_xlabel("j (column)")
    ax2d.set_ylabel("i (row)")

    plt.tight_layout()
    plt.savefig(f"data/plots/surface_heatmap_{N}.png")
    # plt.show()

for N in [128, 256, 512, 1024]:
    Z = load_data(f"data/grids/grid_{N}.bin")
    visualise_surface_and_heatmap(Z)