import numpy as np 

from utils import BLACK, WHITE, GREY1, GREY2, RED, GREEN, BLUE
from utils import EMPTY, E, P1, P2, P1K, P2K


def can_jump (daught1, daught2):
    """ Verify if daught1 can eat daught2 """
    # Same team A
    if daught1 in (P1, P1K) and daught2 in (P1, P1K):
        return False 

    # Same team B
    if daught1 in (P2, P2K) and daught2 in (P2, P2K):
        return False 
    
    # Daught cannot eat king daught
    if (daught1 == P1 and daught2 == P2K) or (daught1 == P2 and daught2 == P1K):
        return False
    
    return True


def verify_move(grid, grid_colors, old_pos, new_pos):
    """ Verify if a movement if possible """
    # Dummy response for the moment 
    return grid[new_pos] == EMPTY #and np.array_equal(grid_colors[new_pos], BLACK)



def ai_step ():
    pass