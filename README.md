# Game of Life with Custom Patterns and GPU Acceleration

This project simulates Conway's Game of Life using Python with the Pygame library for visualization, CUDA (via CuPy) for GPU-accelerated computation, and custom cell aging and color features. The application showcases cellular automata and pattern-based simulations, making it both an educational and visually stimulating exploration of emergent behavior in complex systems.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Code Structure](#code-structure)
5. [Game Rules and Patterns](#game-rules-and-patterns)
6. [Usage](#usage)
7. [Keyboard Controls](#keyboard-controls)
8. [Customization and Extensions](#customization-and-extensions)
9. [Known Issues and Limitations](#known-issues-and-limitations)

---

## Features

- **GPU Acceleration**: Leverages CUDA for faster cell update processing using CuPy.
- **Pattern Support**: Integrates several predefined patterns and Galaga-inspired spaceship patterns.
- **Customizable Colors and Cell Aging**: Cells age and change color over time, with customizable brightness.
- **Interactive Pygame UI**: Start, stop, and reset simulations or add cells by interacting with the Pygame window.

---

## Prerequisites

Ensure you have the following libraries and a CUDA-enabled GPU for optimal performance:

- Python 3.8 or later
- Pygame
- NumPy
- CuPy (with CUDA support)
- SciPy

Install prerequisites with:

```sh
pip install pygame numpy cupy scipy
```

## Installation

1. Clone the repository to your local machine.
2. Ensure all required libraries are installed (see prerequisites).
3. Run the application:

```sh
python main.py
```

## Code Structure

The code is structured into sections for initialization, pattern definition, and cell update logic. Below is a breakdown:

1. **Pygame Initialization**: Sets up the display for visualizing the grid and cell updates.
2. **CUDA Kernel Setup**: Uses a custom CUDA kernel for optimized, parallelized cell updates.
3. **Pattern Definitions**: Contains custom cellular patterns, including oscillator and spaceship structures.
4. **Main Game Loop**: Handles UI events, updates grid states, and renders the grid.

## Game Rules and Patterns

The Game of Life operates on a grid of cells where each cell has two possible states: alive or dead. Cells change states based on the following rules:

1. **Survival**: A live cell with 2 or 3 live neighbors stays alive.
2. **Birth**: A dead cell with exactly 3 live neighbors becomes alive.
3. **Death**: Cells outside these conditions die or remain dead.

### Predefined Patterns

Several patterns are included, each with unique behaviors and movement:
- **1-9**: Classic oscillators and spaceships like the R-pentomino, Diehard, and Acorn.
- **A-E**: Galaga-inspired spaceship patterns.
- **0**: Randomized pattern within a 10x10 grid area.

## Usage

### Adding Cells and Patterns

To add cells, simply click and drag on the Pygame window. Patterns can be placed at the mouse position using predefined keys (0-9 and A-E).

### Running the Simulation

1. **Start**: By default, the simulation is paused on startup. Press **SPACE** to toggle play/pause.
2. **Reset**: Press **R** to clear the grid and start over.
3. **Pattern Selection**: Press number keys (0-9) or letter keys (A-E) to add predefined patterns at the mouse location.

## Keyboard Controls

- **Space**: Toggle play/pause for the simulation.
- **R**: Reset the simulation, clearing all cells.
- **Number keys (0-9)**: Place corresponding predefined patterns at the current mouse position.
- **Letter keys (A-E)**: Place Galaga-inspired patterns at the mouse position.

## Customization and Extensions

### Colors and Age-Based Fading

Cell colors are defined using HSL for vibrant, customizable hues. The function `get_random_bright_color()` provides randomly selected bright colors for each cell. Color fades over time as the cell ages, using a linear interpolation from bright to dark based on a pre-calculated gradient.

### Adding New Patterns

To add new patterns:
1. Update the `PATTERNS` dictionary with the desired cell coordinates.
2. (Optional) Define colors or unique behaviors for new patterns.

### Additional CUDA Kernels

To expand processing capabilities, consider adding more CUDA kernels for specialized behaviors or multi-grid operations.

## Known Issues and Limitations

1. **CUDA Support Required**: The code uses CuPy, which requires a CUDA-compatible GPU.
2. **Pygame Performance Limits**: For very large grids, Pygame’s rendering may become a bottleneck.
3. **Limited Interactivity**: All pattern placements require manual input. Expanding automated behaviors could improve this.

---

## Conclusion

This Game of Life simulation is a high-performance, interactive implementation designed to visualize cellular automata in a compelling way. With GPU acceleration, unique color handling, and interactive features, it's a great way to explore pattern generation and the potential for growth and decay in cellular structures. Enjoy experimenting with patterns and watch emergent behaviors unfold!
