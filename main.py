import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

N = 200  # 200x200 grid (smaller for smoother animation)
spin_ratio = 0.75  # ratio of down-to-up spins

# Initialize lattice
init_random = np.random.random((N, N))
lattice = np.zeros((N, N))
lattice[init_random >= spin_ratio] = 1
lattice[init_random < spin_ratio] = -1

# Custom monochrome colormap
colors = ['#2D2D2D', '#F5F5F0']  # charcoal for -1, off-white for +1
cmap_mono = ListedColormap(colors)

# Metropolis step function (single sweep)
@numba.njit(nopython=True, nogil=True)
def metropolis_step(spin_arr, BJ):
    n_rows, n_cols = spin_arr.shape

    for _ in range(n_rows * n_cols):
        i = np.random.randint(0, n_rows)
        j = np.random.randint(0, n_cols)

        neighbor_sum = (
            spin_arr[(i + 1) % n_rows, j] +
            spin_arr[(i - 1) % n_rows, j] +
            spin_arr[i, (j + 1) % n_cols] +
            spin_arr[i, (j - 1) % n_cols]
        )

        dE = 2 * spin_arr[i, j] * neighbor_sum

        if dE <= 0 or np.random.random() < np.exp(-BJ * dE):
            spin_arr[i, j] *= -1

    return spin_arr

# Calculate magnetization
def get_magnetization(spin_arr):
    return np.abs(np.sum(spin_arr)) / (spin_arr.shape[0] * spin_arr.shape[1])

if __name__ == "__main__":
    BJ_init = 0.45  # Near critical temperature (Tc ≈ 2.269, so BJ_c ≈ 0.44)
    frames = 2000  # Number of animation frames

    # Setup figure with room for slider at bottom
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.15], width_ratios=[1, 1.2])

    # Lattice view
    ax_lattice = fig.add_subplot(gs[0, 0])
    im = ax_lattice.imshow(lattice, cmap=cmap_mono, vmin=-1, vmax=1, interpolation='nearest')
    ax_lattice.set_title(f'Ising Model: {N}×{N} Lattice', fontsize=12)
    ax_lattice.axis('off')

    # Magnetization plot
    ax_mag = fig.add_subplot(gs[0, 1])
    mag_history = []
    line, = ax_mag.plot([], [], 'b-', linewidth=2)
    ax_mag.set_xlim(0, frames)
    ax_mag.set_ylim(0, 1)
    ax_mag.set_xlabel('Monte Carlo Sweeps')
    ax_mag.set_ylabel('Magnetization |M|')
    ax_mag.set_title('Magnetization vs Time')
    ax_mag.grid(True, alpha=0.3)

    # Text for current magnetization and BJ
    info_text = ax_mag.text(0.02, 0.95, '', transform=ax_mag.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Slider axis
    ax_slider = fig.add_subplot(gs[1, :])
    slider = Slider(
        ax=ax_slider,
        label='BJ (βJ) ',
        valmin=0.1,
        valmax=2.0,
        valinit=BJ_init,
        valstep=0.01,
    )

    # Mark critical temperature on slider
    ax_slider.axvline(0.44, color='r', linestyle='--', alpha=0.5)
    ax_slider.text(0.44, 0.5, ' Tc', transform=ax_slider.get_xaxis_transform(),
                   color='r', fontsize=8, verticalalignment='center')

    print(f"Starting simulation with initial BJ = {BJ_init}...")
    print("Use the slider to adjust temperature in real-time!")
    print("Close the window to stop the animation.")

    def update(frame):
        global lattice

        # Get current BJ from slider
        BJ = slider.val

        # Perform one Metropolis sweep
        lattice = metropolis_step(lattice, BJ)

        # Update lattice image
        im.set_data(lattice)

        # Update magnetization
        mag = get_magnetization(lattice)
        mag_history.append(mag)

        # Update plot
        line.set_data(range(len(mag_history)), mag_history)
        info_text.set_text(f'Sweep: {frame}\n|M|: {mag:.3f}\nBJ: {BJ:.2f}')

        return im, line, info_text

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    print("Simulation complete!")
