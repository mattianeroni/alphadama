import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as functional

from utils import EMPTY, E, P1, P2, P1K, P2K

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
def encode (grid):
    """ 
    Encode the state for the neural network.
    :param grid: (B, 8, 8)
    :return: (B, 4, 8, 8)

    The input is encoded in the form (B, 4, 8, 8) where B is the number of batches, 
    8 is the size of the chessboard, and 4 are the kinds of checkers on the chessboard.
    The input matrix is binary --i.e., one where the considered checker is present, zero
    where it isn't.
    """
    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    batch_size = grid.shape[0]
    p1_tensor = torch.where(grid == P1, 1, 0)
    p2_tensor = torch.where(grid == P2, 1, 0)
    p1k_tensor = torch.where(grid == P1K, 1, 0)
    p2k_tensor = torch.where(grid == P2K, 1, 0)
    input_tensor = torch.zeros((batch_size, 4, 8, 8))
    input_tensor[:, 0, :, :] = p1_tensor
    input_tensor[:, 1, :, :] = p2_tensor
    input_tensor[:, 2, :, :] = p1k_tensor
    input_tensor[:, 3, :, :] = p2k_tensor
    return input_tensor 



@torch.no_grad() 
def feasible_moves (grid):
    """ 
    Check all the feasible movements returning a binary matrix.
    :param grid: (B, 8, 8)
    :return: (B, 4, 8, 8), where 4 are the possible directions
    """
    # Left - ahead
    feasibles_left_ahead = torch.zeros(grid.shape) 
    shift_grid = torch.full(grid.shape, np.nan)
    shift_grid[:, :-1, 1:] = grid[:, 1:, :-1]
    double_shift_grid = torch.full(grid.shape, np.nan)
    double_shift_grid[:, :-2, 2:] = grid[:, 2:, :-2]
    feas_coords = torch.where(
        ((grid == P1) | (grid == P1K)) &
        ((shift_grid == EMPTY) | ((torch.isin(shift_grid, torch.tensor([P2, P2K]))) & (double_shift_grid == EMPTY)))
    )
    feasibles_left_ahead[feas_coords] = 1

    # Right - ahead
    feasibles_right_ahead = torch.zeros(grid.shape) 
    shift_grid = torch.full(grid.shape, np.nan)
    shift_grid[:, :-1, :-1] = grid[:, 1:, 1:]
    double_shift_grid = torch.full(grid.shape, np.nan)
    double_shift_grid[:, :-2, :-2] = grid[:, 2:, 2:]
    feas_coords = torch.where(
        ((grid == P1) | (grid == P1K)) &
        ((shift_grid == EMPTY) | ((torch.isin(shift_grid, torch.tensor([P2, P2K]))) & (double_shift_grid == EMPTY)))
    )
    feasibles_right_ahead[feas_coords] = 1
    
    # Left - back 
    feasibles_left_back = torch.zeros(grid.shape)
    shift_grid = torch.full(grid.shape, np.nan)
    shift_grid[:, 1:, 1:] = grid[:, :-1, :-1]
    double_shift_grid = torch.full(grid.shape, np.nan)
    double_shift_grid[:, 2:, 2:] = grid[:, :-2, :-2]
    feas_coords = torch.where(
        (grid == P1K) &
        ((shift_grid == EMPTY) | ((torch.isin(shift_grid, torch.tensor([P2, P2K]))) & (double_shift_grid == EMPTY)))
    )
    feasibles_left_back[feas_coords] = 1
    
    # Right - back
    feasibles_right_back = torch.zeros(grid.shape)
    shift_grid = torch.full(grid.shape, np.nan)
    shift_grid[:, 1:, :-1] = grid[:, :-1, 1:]
    double_shift_grid = torch.full(grid.shape, np.nan)
    double_shift_grid[:, 2:, :-2] = grid[:, :-2, 2:]
    feas_coords = torch.where(
        (grid == P1K) &
        ((shift_grid == EMPTY) | ((torch.isin(shift_grid, torch.tensor([P2, P2K]))) & (double_shift_grid == EMPTY)))
    )
    feasibles_right_back[feas_coords] = 1

    # binary matrix with all the feasible moves
    feasible_moves = torch.zeros((grid.shape[0], 4, 8, 8))
    feasible_moves[:, 0, :, :] = feasibles_left_ahead
    feasible_moves[:, 1, :, :] = feasibles_right_ahead
    feasible_moves[:, 2, :, :] = feasibles_left_back
    feasible_moves[:, 3, :, :] = feasibles_right_back
    return feasible_moves


@torch.no_grad()
def decode (grid, output):
    """ 
    Decode the output of the neural network into a decision.
    :param grid: (B, 8, 8)
    :param output: B tuples like (i, x1, x2) representing the suggestion
                   to move draught in position (x1, x2) in direction i. 

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
    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    feas = feasible_moves(grid).type(torch.float)
    feasible_scores = output * feas
    indexes = np.unravel_index(feasible_scores.flatten(-3).argmax(dim=1), (4, 8, 8))
    return tuple(zip(*indexes))


def save_model (model):
    """ Method to save the model """
    if not os.path.exists("./model"):
        os.makedirs("./model")
    torch.save(model.state_dict(), "./model/model.pth")


def load_model (path="./model/model.pth"):
    """ Method to load the model """
    model = torch.load(path)
    model.eval()
    return model


class DamaModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(5, 5), 
            stride=1, padding=2, dilation=1, groups=1, bias=True, 
            padding_mode='zeros')
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(1, 1), 
            stride=1, padding=2, dilation=1, groups=1, bias=True, 
            padding_mode='zeros')
        self.conv3 = nn.Conv2d(8, 4, kernel_size=(5, 5), 
            stride=1, padding=0, dilation=1, groups=1, bias=True, 
            padding_mode='zeros')

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        return x
