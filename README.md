# Hypergraph Ising Model

A 2D Ising model simulation extended to work with hypergraphs instead of just nearest-neighbor pairwise interactions.

## Overview

In the classical 2D Ising model, each spin interacts only with its 4 nearest neighbors. This implementation generalizes the model to use **hyperedges** - where a single interaction can connect multiple spins simultaneously, even if they're not spatial neighbors.

## Energy Equation

For the hypergraph Ising model, the Hamiltonian is:

$$E = -J \sum_{e} \left( \prod_{i \in e} \sigma_i \right)$$

Where:
- The sum is over all hyperedges $e$
- The product is over **all** spins in each hyperedge
- Each $\sigma_i \in \{+1, -1\}$

This is fundamentally different from pairwise interactions - a hyperedge $\{A, B, C\}$ contributes energy based on the product $\sigma_A \cdot \sigma_B \cdot \sigma_C$, not the sum of pairwise products.

## Features

- **200×200 lattice** with periodic boundary conditions
- **Hypergraph structure**: Spatial neighbors (radius R=1) + random long-range hyperedges
- **Variable hyperedge sizes**: 2 to 10 nodes per hyperedge
- **Real-time visualization** with temperature slider
- **Numba-optimized** for performance

## Installation

```bash
git clone <repo-url>
cd ising_model_2d_hypergraph
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib numba
```

## Usage

```bash
source venv/bin/activate
python hypergraph_ising.py
```

Use the slider to adjust the temperature (βJ) in real-time.

## Configuration

Key parameters in `hypergraph_ising.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 200 | Grid size |
| `MAX_HYPEREDGE_SIZE` | 10 | Maximum nodes per hyperedge |
| `MIN_HYPEREDGE_SIZE` | 2 | Minimum nodes per hyperedge |
| `RADIUS` | 1 | Spatial radius for local connections |
| `HYPEREDGES_PER_NODE` | 4 | Hyperedges centered at each node |
| `LONG_RANGE_PROBABILITY` | 0.05 | Probability of long-range hyperedge |

## Files

- `hypergraph_ising.py` - Main hypergraph simulation
- `main.py` - Original 2D nearest-neighbor Ising model (for comparison)
- `notes.md` - Theoretical background

## Physics Background

The Metropolis algorithm is used to sample equilibrium states. For each spin flip, the energy change is:

$$\Delta E = 2J \cdot \sigma_i \cdot \sum_{e \ni i} \left( \prod_{j \in e \setminus \{i\}} \sigma_j \right)$$

The flip is accepted if $\Delta E \leq 0$ or with probability $e^{-\beta \Delta E}$.
