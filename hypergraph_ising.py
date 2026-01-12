import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from typing import List, Tuple

# ==========================================
# CONFIGURATION
# ==========================================
N = 200  # Grid size
N_SQUARED = N * N
SPIN_RATIO = 0.75  # Ratio of down-to-up spins

# Hypergraph parameters
RADIUS = 1  # Spatial radius for local connections
MAX_HYPEREDGE_SIZE = 10  # Maximum size of a hyperedge
MIN_HYPEREDGE_SIZE = 2  # Minimum size of a hyperedge
HYPEREDGES_PER_NODE = 4  # Average number of hyperedges each node participates in
LONG_RANGE_PROBABILITY = 0.05  # Probability of a long-range hyperedge per local hyperedge
J_COUPLING = 1.0  # Coupling constant

# ==========================================
# HYPERGRAPH GENERATION
# ==========================================

def get_spatial_neighbors(node_idx: int, radius: int) -> np.ndarray:
    """
    Get all nodes within given radius (Chebyshev distance) on the 2D toroidal grid.
    Returns array of node indices (excluding the node itself).
    """
    i, j = node_idx // N, node_idx % N
    neighbors = []

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            # Periodic boundary conditions
            ni = (i + di) % N
            nj = (j + dj) % N
            neighbors.append(ni * N + nj)

    return np.array(neighbors, dtype=np.int32)


def generate_hypergraph() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate hyperedges for the Ising model.

    Returns:
        hyperedges: List of arrays, each containing node indices in that hyperedge
        node_to_hyperedges: List of arrays, mapping each node to hyperedge indices it belongs to
    """
    print(f"Generating hypergraph with {N_SQUARED} nodes...")
    print(f"  Radius R = {RADIUS}")
    print(f"  Hyperedge sizes: {MIN_HYPEREDGE_SIZE} to {MAX_HYPEREDGE_SIZE}")
    print(f"  ~{HYPEREDGES_PER_NODE} hyperedges per node")

    hyperedges = []
    node_to_hyperedges = [[] for _ in range(N_SQUARED)]

    # Generate hyperedges centered at each node
    for node in range(N_SQUARED):
        neighbors = get_spatial_neighbors(node, RADIUS)

        # Create local hyperedges
        num_local = HYPEREDGES_PER_NODE
        for _ in range(num_local):
            # Random hyperedge size
            size = np.random.randint(MIN_HYPEREDGE_SIZE, min(MAX_HYPEREDGE_SIZE, len(neighbors) + 2))

            # Decide if this is a long-range hyperedge
            is_long_range = np.random.random() < LONG_RANGE_PROBABILITY

            if is_long_range:
                # Pick random nodes from anywhere in the lattice
                all_nodes = np.arange(N_SQUARED)
                selected = np.random.choice(all_nodes, size=size - 1, replace=False)
            else:
                # Pick from local neighbors
                num_from_local = min(size - 1, len(neighbors))
                selected = np.random.choice(neighbors, size=num_from_local, replace=False)

                # If need more nodes, add random long-range ones
                if num_from_local < size - 1:
                    remaining = size - 1 - num_from_local
                    all_nodes = np.arange(N_SQUARED)
                    # Exclude already selected and the center node
                    available = np.setdiff1d(all_nodes, np.append(selected, node))
                    long_range = np.random.choice(available, size=remaining, replace=False)
                    selected = np.append(selected, long_range)

            # Create hyperedge with center node + selected nodes
            hyperedge = np.append(node, selected).astype(np.int32)
            hyperedge_idx = len(hyperedges)
            hyperedges.append(hyperedge)

            # Update node_to_hyperedges mapping
            for n in hyperedge:
                node_to_hyperedges[n].append(hyperedge_idx)

    # Convert to numpy array of arrays for Numba compatibility
    # We'll use a flat representation for Numba
    node_to_hyperedges_flat = []
    node_to_hyperedges_ptrs = [0]

    for lst in node_to_hyperedges:
        node_to_hyperedges_flat.extend(lst)
        node_to_hyperedges_ptrs.append(len(node_to_hyperedges_flat))

    print(f"  Generated {len(hyperedges)} total hyperedges")
    print(f"  Average hyperedges per node: {len(node_to_hyperedges_flat) / N_SQUARED:.2f}")

    return hyperedges, np.array(node_to_hyperedges_flat, dtype=np.int32), np.array(node_to_hyperedges_ptrs, dtype=np.int32)


# ==========================================
# NUMBA OPTIMIZED DATA STRUCTURES
# ==========================================

def prepare_numba_structures(hyperedges: List[np.ndarray],
                             node_to_hyperedges_flat: np.ndarray,
                             node_to_hyperedges_ptrs: np.ndarray) -> Tuple:
    """
    Convert hypergraph data to Numba-friendly format.

    Returns:
        hyperedge_data: Flat array of all hyperedge node indices
        hyperedge_ptrs: Pointers to start of each hyperedge in hyperedge_data
        hyperedge_sizes: Size of each hyperedge
        node_to_hyperedges_flat: Already in flat format
        node_to_hyperedges_ptrs: Already in flat format
    """
    # Flatten hyperedges into single array
    hyperedge_data = []
    hyperedge_sizes = []
    hyperedge_ptrs = [0]

    for he in hyperedges:
        hyperedge_data.extend(he)
        hyperedge_sizes.append(len(he))
        hyperedge_ptrs.append(len(hyperedge_data))

    return (
        np.array(hyperedge_data, dtype=np.int32),
        np.array(hyperedge_ptrs, dtype=np.int32),
        np.array(hyperedge_sizes, dtype=np.int32),
        node_to_hyperedges_flat,
        node_to_hyperedges_ptrs
    )


# ==========================================
# METROPOLIS ALGORITHM FOR HYPERGRAPH
# ==========================================

@njit(nopython=True, nogil=True)
def metropolis_step_hypergraph(
    spins: np.ndarray,
    BJ: float,
    hyperedge_data: np.ndarray,
    hyperedge_ptrs: np.ndarray,
    hyperedge_sizes: np.ndarray,
    node_to_hyperedges_flat: np.ndarray,
    node_to_hyperedges_ptrs: np.ndarray
) -> np.ndarray:
    """
    Perform one Metropolis sweep for the hypergraph Ising model.

    Energy: E = -J * sum_e (product of all spins in hyperedge e)
    """
    n_spins = len(spins)

    for _ in range(n_spins):
        # Pick random spin
        i = np.random.randint(0, n_spins)

        # Calculate energy change if we flip spin i
        # dE = 2 * J * spins[i] * sum_{e contains i} (product of other spins in e)
        hyperedge_product_sum = 0.0

        # Get hyperedges containing node i
        start_idx = node_to_hyperedges_ptrs[i]
        end_idx = node_to_hyperedges_ptrs[i + 1]

        for he_idx in range(start_idx, end_idx):
            e = node_to_hyperedges_flat[he_idx]

            # Get hyperedge data
            he_start = hyperedge_ptrs[e]
            he_end = hyperedge_ptrs[e + 1]
            he_size = hyperedge_sizes[e]

            # Compute product of all spins in hyperedge except i
            product = 1.0
            for k in range(he_start, he_end):
                node = hyperedge_data[k]
                if node != i:
                    product *= spins[node]

            hyperedge_product_sum += product

        dE = 2.0 * J_COUPLING * spins[i] * hyperedge_product_sum

        # Metropolis acceptance criterion
        if dE <= 0 or np.random.random() < np.exp(-BJ * dE):
            spins[i] *= -1

    return spins


@njit(nopython=True, nogil=True)
def compute_total_energy(
    spins: np.ndarray,
    hyperedge_data: np.ndarray,
    hyperedge_ptrs: np.ndarray,
    hyperedge_sizes: np.ndarray
) -> float:
    """
    Compute total energy of the system.

    E = -J * sum_e (product of all spins in hyperedge e)
    """
    total_energy = 0.0
    num_hyperedges = len(hyperedge_sizes)

    for e in range(num_hyperedges):
        he_start = hyperedge_ptrs[e]
        he_end = hyperedge_ptrs[e + 1]

        # Compute product of all spins in this hyperedge
        product = 1.0
        for k in range(he_start, he_end):
            node = hyperedge_data[k]
            product *= spins[node]

        total_energy += product

    return -J_COUPLING * total_energy


# ==========================================
# VISUALIZATION AND MAIN LOOP
# ==========================================

def get_magnetization(spin_arr: np.ndarray) -> float:
    return np.abs(np.sum(spin_arr)) / len(spin_arr)


def reshape_to_2d(spins_1d: np.ndarray) -> np.ndarray:
    return spins_1d.reshape((N, N))


def main():
    # Initialize lattice (1D for hypergraph, but we'll reshape for visualization)
    init_random = np.random.random(N_SQUARED)
    spins = np.zeros(N_SQUARED)
    spins[init_random >= SPIN_RATIO] = 1
    spins[init_random < SPIN_RATIO] = -1

    # Generate hypergraph structure
    hyperedges, node_he_flat, node_he_ptrs = generate_hypergraph()

    # Prepare Numba-friendly structures
    hyperedge_data, hyperedge_ptrs, hyperedge_sizes, node_he_flat, node_he_ptrs = prepare_numba_structures(
        hyperedges, node_he_flat, node_he_ptrs
    )

    BJ_init = 0.5  # Near critical temperature
    frames = 2000

    # Custom monochrome colormap
    colors = ['#2D2D2D', '#F5F5F0']
    cmap_mono = ListedColormap(colors)

    # Setup figure
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.15], width_ratios=[1, 1.2])

    # Lattice view
    ax_lattice = fig.add_subplot(gs[0, 0])
    lattice_2d = reshape_to_2d(spins)
    im = ax_lattice.imshow(lattice_2d, cmap=cmap_mono, vmin=-1, vmax=1, interpolation='nearest')
    ax_lattice.set_title(f'Hypergraph Ising Model: {N}×{N} Lattice\n{len(hyperedges)} hyperedges, max size {MAX_HYPEREDGE_SIZE}', fontsize=12)
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

    # Info text
    info_text = ax_mag.text(0.02, 0.95, '', transform=ax_mag.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Slider
    ax_slider = fig.add_subplot(gs[1, :])
    slider = Slider(
        ax=ax_slider,
        label='BJ (βJ) ',
        valmin=0.1,
        valmax=2.0,
        valinit=BJ_init,
        valstep=0.01,
    )

    print(f"\nStarting simulation with initial BJ = {BJ_init}...")
    print("Use the slider to adjust temperature in real-time!")
    print("Close the window to stop the animation.\n")

    def update(frame):
        nonlocal spins

        BJ = slider.val

        # Perform one Metropolis sweep
        spins = metropolis_step_hypergraph(
            spins, BJ,
            hyperedge_data, hyperedge_ptrs, hyperedge_sizes,
            node_he_flat, node_he_ptrs
        )

        # Update lattice image
        im.set_data(reshape_to_2d(spins))

        # Update magnetization
        mag = get_magnetization(spins)
        mag_history.append(mag)

        # Update plot
        line.set_data(range(len(mag_history)), mag_history)
        info_text.set_text(f'Sweep: {frame}\n|M|: {mag:.3f}\nBJ: {BJ:.2f}')

        return im, line, info_text

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    print("Simulation complete!")


if __name__ == "__main__":
    main()
