import torch 
import numpy as np 

from utils import EMPTY, E, P1, P2, P1K, P2K
import draughts



def encode (grid):
    """ Encode the state for the neural network """
    p1_tensor = np.where(grid == P1, 1, 0)
    p2_tensor = np.where(grid == P2, 1, 0)
    p1k_tensor = np.where(grid == P1K, 1, 0)
    p2k_tensor = np.where(grid == P2K, 1, 0)
    input_tensor = np.stack((p1_tensor, p2_tensor, p1k_tensor, p2k_tensor), axis=0)
    #input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1)
    return torch.unsqueeze(input_tensor, 0)


def decode (grid, output):
    """ 
    Decode the output of the neural network into a decision.
    Initially we encode the output as a single matrix 4x8x8 where 
    element (i, x, y) represent the movement of daught in position
    (x, y) in direction i.
    See how the ouput only contains information about the direction, 
    if the movement will be of one cell or two cells depends on the 
    context.
    Positions i are mapped into the output in the following order:
        - 0: left ahead
        - 1: right ahead 
        - 2: left back
        - 3: right back 

    NOTE: We could consider to split the output in a matrix 8x8 
    indicating the daught to move, and a softmax defining the direction.
    In this case we would have less parameters to train.
    """
    # Left - ahead
    feasibles_left_ahead = np.zeros((8, 8))
    shift_grid = np.full((8, 8), np.nan)
    shift_grid[:-1, 1:] = grid[1:, :-1]
    double_shift_grid = np.full((8, 8), np.nan)
    double_shift_grid[:-2, 2:] = grid[2:, :-2]
    feas_coords = np.where(
        ((grid == P1) | (grid == P1K)) &
        ((shift_grid == EMPTY) | ((np.isin(shift_grid, (P2, P2K))) & (double_shift_grid == EMPTY)))
    )
    feasibles_left_ahead[feas_coords] = 1

    # Right - ahead
    feasibles_right_ahead = np.zeros((8, 8))
    shift_grid = np.full((8, 8), np.nan)
    shift_grid[:-1, :-1] = grid[1:, 1:]
    double_shift_grid = np.full((8, 8), np.nan)
    double_shift_grid[:-2, :-2] = grid[2:, 2:]
    feas_coords = np.where(
        ((grid == P1) | (grid == P1K)) &
        ((shift_grid == EMPTY) | ((np.isin(shift_grid, (P2, P2K))) & (double_shift_grid == EMPTY)))
    )
    feasibles_right_ahead[feas_coords] = 1
    
    # Left - back 
    feasibles_left_back = np.zeros((8, 8))
    shift_grid = np.full((8, 8), np.nan)
    shift_grid[1:, 1:] = grid[:-1, :-1]
    double_shift_grid = np.full((8, 8), np.nan)
    double_shift_grid[2:, 2:] = grid[:-2, :-2]
    feas_coords = np.where(
        (grid == P1K) &
        ((shift_grid == EMPTY) | ((np.isin(shift_grid, (P2, P2K))) & (double_shift_grid == EMPTY)))
    )
    feasibles_left_back[feas_coords] = 1
    
    # Right - back
    feasibles_right_back = np.zeros((8, 8))
    shift_grid = np.full((8, 8), np.nan)
    shift_grid[1:, :-1] = grid[:-1, 1:]
    double_shift_grid = np.full((8, 8), np.nan)
    double_shift_grid[2:, :-2] = grid[:-2, 2:]
    feas_coords = np.where(
        (grid == P1K) &
        ((shift_grid == EMPTY) | ((np.isin(shift_grid, (P2, P2K))) & (double_shift_grid == EMPTY)))
    )
    feasibles_right_back[feas_coords] = 1

    # Filter the output keeping only the scores corresponding to feasible movements 
    feas = np.stack((feasibles_left_ahead, feasibles_right_ahead, feasibles_left_back, feasibles_right_back), axis=0)
    feasible_scores = output * feas
    y = np.unravel_index(feasible_scores.argmax(), feasible_scores.shape)
    return y