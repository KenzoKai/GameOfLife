import pygame
import numpy as np
import cupy as cp
from scipy.ndimage import label as scipy_label, distance_transform_edt
import colorsys
import random

# Initialize Pygame and create display
pygame.init()

# Set grid dimensions first, then calculate screen size
CELL_SIZE = 5
GRID_WIDTH = 350  # This will create a 2560-pixel wide screen (512 * 5)
GRID_HEIGHT = 350  # This will create a 1440-pixel high screen (288 * 5)
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE

print(f"Grid dimensions: {GRID_WIDTH}x{GRID_HEIGHT}")
print(f"Screen dimensions: {WIDTH}x{HEIGHT}")

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
surface = pygame.Surface((WIDTH, HEIGHT))
pygame.display.set_caption("The Game of Life")

# Initialize empty grid with correct dimensions
grid = cp.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=cp.int32)

# Add age tracking to the grid
cell_ages = cp.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=cp.int32)

# Add color tracking to the grid
cell_colors = {}  # Dictionary to store colors for each cell

def screen_to_grid(pos):
    """Convert screen coordinates to grid coordinates"""
    x = pos[0] // CELL_SIZE
    y = pos[1] // CELL_SIZE
    return (
        max(0, min(x, GRID_WIDTH - 1)),
        max(0, min(y, GRID_HEIGHT - 1))
    )

def draw_at_position(grid_np, ages_np, pos):
    """Draw live cells at the given position"""
    x, y = screen_to_grid(pos)
    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
        grid_np[x, y] = 1
        ages_np[x, y] = 0  # Reset age for new cells
    return grid_np, ages_np

# CUDA kernel with corrected indexing
update_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_grid(const int* grid, int* new_grid, int* ages, int* new_ages, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int total = 0;
    
    // Count live neighbors
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            
            int ni = (x + di + width) % width;
            int nj = (y + dj + height) % height;
            int nidx = nj * width + ni;
            
            total += grid[nidx];
        }
    }
    
    // Apply Game of Life rules and update ages
    if (grid[idx] == 1) {
        if (total == 2 || total == 3) {
            new_grid[idx] = 1;
            new_ages[idx] = ages[idx] + 1;  // Increment age for surviving cells
        } else {
            new_grid[idx] = 0;
            new_ages[idx] = 0;
        }
    } else {
        if (total == 3) {
            new_grid[idx] = 1;
            new_ages[idx] = 0;  // New cells start with age 0
        } else {
            new_grid[idx] = 0;
            new_ages[idx] = 0;
        }
    }
}
''', 'update_grid')

def create_density_gradient():
    """Create a gradient from light grey to dark grey"""
    colors = []
    for i in range(100):
        # Start from lighter grey (128, 128, 128) to darker grey (40, 40, 40)
        ratio = 1 - (i / 99)
        grey_value = int(40 + (88 * ratio))  # 128 -> 40
        colors.append((grey_value, grey_value, grey_value))
    return colors

# Pre-calculate density gradient colors
density_colors = create_density_gradient()

def calculate_density_color(total_population, max_allowed):
    """Calculate color based on population density"""
    density_ratio = min(total_population / max_allowed, 1.0)
    color_idx = int(density_ratio * 99)
    return density_colors[color_idx]

def draw_grid(grid_gpu, ages_gpu, surface):
    grid = cp.asnumpy(grid_gpu)
    ages = cp.asnumpy(ages_gpu)
    surface.fill((0, 0, 0))  # Black background
    
    # Calculate current population and density
    total_population = int(cp.sum(grid_gpu))
    max_allowed = int(GRID_WIDTH * GRID_HEIGHT * 0.20)  # 20% of grid
    base_color = calculate_density_color(total_population, max_allowed)
    
    rect = pygame.Rect(0, 0, CELL_SIZE - 1, CELL_SIZE - 1)
    
    # Draw all live cells
    live_positions = np.where(grid == 1)
    for x, y in zip(*live_positions):
        rect.topleft = (x * CELL_SIZE, y * CELL_SIZE)
        
        if ages[x, y] < 3:  # New cells flash blue
            # Fade from bright blue to the density-based grey
            fade_ratio = ages[x, y] / 3
            start_color = (100, 150, 255)  # Bright blue
            
            # Interpolate between blue and density-based grey
            current_color = tuple(
                int(start + (end - start) * fade_ratio)
                for start, end in zip(start_color, base_color)
            )
        else:
            current_color = base_color
        
        pygame.draw.rect(surface, current_color, rect)
    
    screen.blit(surface, (0, 0))
    pygame.display.flip()

# CUDA grid dimensions
block_size = (16, 16)
grid_size = ((GRID_WIDTH + block_size[0] - 1) // block_size[0],
             (GRID_HEIGHT + block_size[1] - 1) // block_size[1])

# Add pattern definitions
PATTERNS = {
    # 1: Pentadecathlon (period 15 oscillator)
    '1': [
        (0, -1), (0, 0), (0, 1),  # Center column
        (0, -4), (0, -3), (0, -2),  # Top section
        (0, 2), (0, 3), (0, 4),  # Bottom section
        (-1, -3), (1, -3),  # Top stabilizers
        (-1, 3), (1, 3)  # Bottom stabilizers
    ],
    
    # 2: Cross (period 3 oscillator)
    '2': [
        # Center cross
        (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        # Corner stabilizers
        (-2, -2), (-2, 2), (2, -2), (2, 2)
    ],
    
    # 3: Figure Eight (period 8 oscillator)
    '3': [
        # Left square
        (-2, -1), (-2, 0), (-1, -1), (-1, 0),
        # Right square
        (1, 0), (1, 1), (2, 0), (2, 1),
        # Connectors
        (0, -2), (0, 2)
    ],
    
    # 4: Linear Breeder (grows continuously)
    '4': [
        # Base LWSS generator
        (0, 0), (1, 0), (2, 0), (3, 0),
        (-1, 1), (3, 1),
        (3, 2),
        (-1, 3), (2, 3),
        
        # Reaction blocks (these create new spaceships)
        (8, 0), (8, 1),
        (8, 3), (8, 4),
        
        # Stabilizer blocks
        (18, 0), (19, 0), (18, 1), (19, 1),
        (18, 3), (19, 3), (18, 4), (19, 4),
        
        # Additional reaction blocks for sustained growth
        (28, 0), (28, 1),
        (28, 3), (28, 4),
        
        # Final stabilizer blocks
        (38, 0), (39, 0), (38, 1), (39, 1),
        (38, 3), (39, 3), (38, 4), (39, 4)
    ],
    
    # 5: R-pentomino (chaotic pattern)
    '5': [(0, 0), (1, 0), (-1, 1), (0, 1), (0, 2)],
    
    # 6: Diehard (disappears after 130 generations)
    '6': [(0, 0), (1, 0), (1, 1), (5, 1), (6, 1), (7, 1), (6, -1)],
    
    # 7: Acorn (grows chaotically)
    '7': [(0, 0), (-1, 1), (1, 1), (2, 1), (3, 1), (4, 1), (-2, -1)],
    
    # 8: Infinite growth pattern
    '8': [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)],
    
    # 9: Lightweight Spaceship (LWSS) fleet
    '9': [(i, j) for i in range(0, 15, 5) for j in range(3) if (i+j) % 2 == 0],
    
    # 0: Random 10x10 block
    '0': [(x, y) for x in range(10) for y in range(10) if random.random() < 0.4]
}

# Add Galaga patterns to the existing PATTERNS dictionary
GALAGA_PATTERNS = {
    # A: Galaga Boss (large butterfly ship)
    'a': [
        # Main body
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (-1, 1), (5, 1),
        (-2, 2), (6, 2),
        (-2, 3), (-1, 3), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
        (-1, 4), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4),
        (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
        # Wings
        (-3, 2), (-4, 3), (-3, 4),
        (7, 2), (8, 3), (7, 4)
    ],

    # B: Basic Fighter (small ship)
    'b': [
        # Main body
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (3, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ],

    # C: Challenging Stage Ship (scorpion)
    'c': [
        # Body
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (3, 1),
        (-2, 2), (4, 2),
        (-1, 2), (0, 2), (1, 2), (2, 2), (3, 2),
        # Tail
        (1, -1), (1, -2),
        (0, -2), (2, -2),
        # Claws
        (-2, 1), (4, 1)
    ],

    # D: Diving Fighter (bee-like ship)
    'd': [
        # Main body
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (3, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (2, 3),
        # Wings
        (-2, 1), (4, 1)
    ],

    # E: Enterprise-like ship
    'e': [
        # Saucer section
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        # Engineering section
        (0, 1), (0, 2), (0, 3),
        # Nacelles
        (-2, 2), (-1, 2), (1, 2), (2, 2),
        (-2, 3), (-1, 3), (1, 3), (2, 3)
    ]
}

def place_pattern(grid_np, ages_np, pattern_key, pos):
    """Place a predefined pattern at the given position"""
    base_x, base_y = screen_to_grid(pos)
    
    if pattern_key not in PATTERNS:
        return grid_np, ages_np
        
    # Check current population
    current_population = np.sum(grid_np)
    max_allowed = int(GRID_WIDTH * GRID_HEIGHT * 0.20)
    
    if current_population >= max_allowed:
        return grid_np, ages_np
    
    pattern = PATTERNS[pattern_key]
    
    # Special handling for pattern 4 (Breeder)
    if pattern_key == '4':
        # Center the pattern and rotate it for better viewing
        base_x = GRID_WIDTH // 2 - 20
        base_y = GRID_HEIGHT // 2 - 3
        
        # Create mirrored version for symmetrical growth
        extended_pattern = []
        for dx, dy in pattern:
            extended_pattern.append((dx, dy))
            extended_pattern.append((-dx - 1, dy))
            extended_pattern.append((dx, -dy - 1))
            extended_pattern.append((-dx - 1, -dy - 1))
        pattern = extended_pattern
    
    # Count new cells
    new_cells = 0
    for dx, dy in pattern:
        x = (base_x + dx) % GRID_WIDTH
        y = (base_y + dy) % GRID_HEIGHT
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and grid_np[x, y] == 0:
            new_cells += 1
    
    # Place pattern if within population limit
    if current_population + new_cells <= max_allowed:
        for dx, dy in pattern:
            x = (base_x + dx) % GRID_WIDTH
            y = (base_y + dy) % GRID_HEIGHT
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                grid_np[x, y] = 1
                ages_np[x, y] = 0  # New cells start with age 0
    
    return grid_np, ages_np

# Main game loop
clock = pygame.time.Clock()
running = True
paused = True  # Start paused
mouse_down = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                grid = cp.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=cp.int32)
                cell_ages = cp.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=cp.int32)
                cell_colors.clear()
            # Handle number keys for existing patterns
            elif event.unicode in '0123456789':
                grid_np = cp.asnumpy(grid)
                ages_np = cp.asnumpy(cell_ages)
                grid_np, ages_np = place_pattern(grid_np, ages_np, event.unicode, pygame.mouse.get_pos())
                grid = cp.array(grid_np)
                cell_ages = cp.array(ages_np)
            # Handle Galaga patterns
            elif event.unicode.lower() in GALAGA_PATTERNS:
                grid_np = cp.asnumpy(grid)
                ages_np = cp.asnumpy(cell_ages)
                pattern = GALAGA_PATTERNS[event.unicode.lower()]
                base_x, base_y = screen_to_grid(pygame.mouse.get_pos())
                
                # Check population limit
                current_population = np.sum(grid_np)
                max_allowed = int(GRID_WIDTH * GRID_HEIGHT * 0.20)
                
                if current_population < max_allowed:
                    for dx, dy in pattern:
                        x = (base_x + dx) % GRID_WIDTH
                        y = (base_y + dy) % GRID_HEIGHT
                        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                            grid_np[x, y] = 1
                            ages_np[x, y] = 0  # New cells start with age 0
                
                grid = cp.array(grid_np)
                cell_ages = cp.array(ages_np)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
            grid_np = cp.asnumpy(grid)
            ages_np = cp.asnumpy(cell_ages)
            grid_np, ages_np = draw_at_position(grid_np, ages_np, pygame.mouse.get_pos())
            grid = cp.array(grid_np)
            cell_ages = cp.array(ages_np)
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
        elif event.type == pygame.MOUSEMOTION and mouse_down:
            grid_np = cp.asnumpy(grid)
            ages_np = cp.asnumpy(cell_ages)
            grid_np, ages_np = draw_at_position(grid_np, ages_np, pygame.mouse.get_pos())
            grid = cp.array(grid_np)
            cell_ages = cp.array(ages_np)

    if not paused:
        new_grid = cp.zeros_like(grid)
        new_ages = cp.zeros_like(cell_ages)
        update_kernel(grid_size, block_size, (grid, new_grid, cell_ages, new_ages, GRID_WIDTH, GRID_HEIGHT))
        grid = new_grid.copy()
        cell_ages = new_ages.copy()
    
    draw_grid(grid, cell_ages, surface)
    clock.tick(60)
    pygame.display.set_caption(f"Game of Life - FPS: {clock.get_fps():.1f}")

pygame.quit()
