import numpy as np 

from utils import BLACK, WHITE, GREY1, GREY2, RED, GREEN, BLUE
from utils import EMPTY, E, P1, P2, P1K, P2K
from utils import manhattan_distance, euclidean_distance


def can_jump (eater, eated):
    """ Verify if eater can eat eated """
    # empty cell 
    if eated == EMPTY:
        return False 

    # Same team A
    if eater in (P1, P1K) and eated in (P1, P1K):
        return False 

    # Same team B
    if eater in (P2, P2K) and eated in (P2, P2K):
        return False 
    
    # deprecated
    # Daught cannot eat king daught
    #if (daught1 == P1 and daught2 == P2K) or (daught1 == P2 and daught2 == P1K):
    #    return False
    return True


def verify_move(grid, grid_colors, old_pos, new_pos):
    """ Verify if a movement is possible """
    # non-empty destination cell or white destination cell
    if grid[new_pos] != EMPTY or np.array_equal(grid_colors[new_pos], WHITE):
        return False 

    # fast way to detect an andesired move
    if manhattan_distance(old_pos, new_pos) > 4 or euclidean_distance(old_pos, new_pos) > 3:
        return False
    
    # user draughts trying to go back
    if grid[old_pos] == P2 and new_pos[0] >= old_pos[0]:
        return False 

    # ai draughts trying to go back 
    if grid[old_pos] == P1 and new_pos[0] <= old_pos[0]:
        return False 

    # a draught is eating a draught of the same team
    mid_pos = (old_pos[0] + (new_pos[0] - old_pos[0]) // 2, old_pos[1] + (new_pos[1] - old_pos[1]) // 2)
    if manhattan_distance(old_pos, new_pos) == 4 and not can_jump(grid[old_pos], grid[mid_pos]):
        return False

    return True


def translate_ai_move (grid, grid_colors, move, verify=True):
    """ 
    Translate a result provided by the agent into a detailed movement. 
    The agent only provides the direction in the following way:
    Positions i are mapped into the output in the following order:
        - 0: left ahead (bottom)
        - 1: right ahead (bottom)
        - 2: left back (up)
        - 3: right back (up)
    If the movement is single or double depends on the context.
    """
    direction, old_pos = move[0], (move[1], move[2])
    sm = {0: (old_pos[0] + 1, old_pos[1] - 1), 1: (old_pos[0] + 1, old_pos[1] + 1), 
          2: (old_pos[0] - 1, old_pos[1] - 1), 3: (old_pos[0] - 1, old_pos[1] + 1)}
    dm = {0: (old_pos[0] + 2, old_pos[1] - 2), 1: (old_pos[0] + 2, old_pos[1] + 2), 
          2: (old_pos[0] - 2, old_pos[1] - 2), 3: (old_pos[0] - 2, old_pos[1] + 2)}
    new_pos = sm[direction] if grid[sm[direction]] == EMPTY else dm[direction]
    if verify and not verify_move(grid, grid_colors, old_pos, new_pos):
        raise Exception(f"Unpossible movement from {old_pos} to {new_pos} selected by the model. Arised from move {move}")
    return old_pos, new_pos


