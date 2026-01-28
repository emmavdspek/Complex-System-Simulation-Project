"""
Docstring for ca_model_lila.py
Basic 2D cellular automaton update rules, and logic 
Implements: 
- grid initialization
- neighbor counting 
- simple update rule 
- one update step
"""

import numpy as np
import matplotlib.pyplot as plt

# %%
def initialize_grid(nx= 100, ny = 100, p_tree = 0.5):
    """
    Functions that creates the grid 
    nx, ny = width and height of grid 
    p_tree = probability a cell starts as a tree (1)
    
    Create a grid of size 100*100 of random numbers between 0 and 1, which are compared to a treshold of 0.5
    If above ->True, if below -> False
    asyptye converts True -> 1, False->0
    """
    return(np.random.rand(nx, ny)<p_tree).astype(int) 

def count_neighbors(grid, x, y):
    """
    function counts how many neighbors around cell are 1 (trees), using peeriodic boundary conditions
    """
    nx, ny = grid.shape 

    up = grid[(x - 1) % nx, y] #the % nx and % ny make the grid wrap around (periodic boundaries)
    down = grid [(x + 1) % nx, y]
    left = grid [x,(y- 1) % ny]
    right = grid[x,(y + 1) % ny]

    return up + down + left + right 

def update_cell(state, n_neighbors):
    """  
    Function which decides what happens to a single cell
    :param state: empty or a tree (1)
    :param n_neighbors: if a cell has 3 or 4 neighbors, it becomes empty, otherwise a tree
    """
    if n_neighbors in (3, 4): 
        return 0 
    else: 
        return 1

def update_grid(grid): 
    """ 
    function updates every cell in the grid (synchronous update)
    Returns a new grid. 
    """ 
    nx, ny = grid.shape 
    new_grid = np.zeros_like(grid) #new grid the same size to store the updated values 
    
    for x in range(nx): 
        for y in range(ny): # loop over every cell in the frid 
            state = grid[x, y] # get the current state (0 or 1)
            n = count_neighbors(grid, x, y) 
            new_grid[x, y] = update_cell(state, n) 
    return new_grid

"""
Visualization
"""
grid = initialize_grid()
plt.imshow(grid, cmap="Greens") 
plt.colorbar() 
plt.title("CA grid") 
plt.show()

for t in range(3):
    plt.imshow(grid, cmap="Greens")
    plt.title(f"Step {t}")
    plt.show()
    grid = update_grid(grid)