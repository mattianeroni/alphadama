import os 
import sys
import pygame
import torch
import numpy as np
import itertools

from utils import BLACK, WHITE, GREY1, GREY2, RED, GREEN, BLUE
from utils import EMPTY, E, P1, P2, P1K, P2K
from utils import manhattan_distance, euclidean_distance
from model import load_model, save_model, DamaModel, encode, decode
from agent import DamaAgent
import draughts


WIDTH, HEIGHT = 500, 500
CELL_WIDTH, CELL_HEIGHT = WIDTH / 8, HEIGHT / 8
FPS = 30

grid = np.array([
    [P1, EMPTY, P1, EMPTY, P1, EMPTY, P1, EMPTY],
    [EMPTY, P1, EMPTY, P1, EMPTY, P1, EMPTY, P1],
    [P1, EMPTY, P1, EMPTY, P1, EMPTY, P1, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, P2, EMPTY, P2, EMPTY, P2, EMPTY, P2],
    [P2, EMPTY, P2, EMPTY, P2, EMPTY, P2, EMPTY],
    [EMPTY, P2, EMPTY, P2, EMPTY, P2, EMPTY, P2]
])
grid_colors = np.array([
    [BLACK, WHITE] * 4,
    [WHITE, BLACK] * 4,
    [BLACK, WHITE] * 4,
    [WHITE, BLACK] * 4,
    [BLACK, WHITE] * 4,
    [WHITE, BLACK] * 4,
    [BLACK, WHITE] * 4,
    [WHITE, BLACK] * 4,
])
moving_draught = None
moving_draught_pos = None
USER_TURN = True

# Init game and game instances
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Images for graphics
crown_img = pygame.transform.scale(pygame.image.load("./static/crown.png"), (CELL_WIDTH / 2, CELL_HEIGHT / 2)) 

# Flag to train the model while playing
TRAIN = True if len(sys.argv) > 1 and sys.argv[1] == "--train" else False

# Load the model 
model = load_model("./model/model.pth") if os.path.exists("./model") else DamaModel()
agent = DamaAgent(model, randomness=0.0)

def reset ():
    global user_turn, grid, moving_draught, moving_draught_pos
    grid = np.array([
        [P1, EMPTY, P1, EMPTY, P1, EMPTY, P1, EMPTY],
        [EMPTY, P1, EMPTY, P1, EMPTY, P1, EMPTY, P1],
        [P1, EMPTY, P1, EMPTY, P1, EMPTY, P1, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, P2, EMPTY, P2, EMPTY, P2, EMPTY, P2],
        [P2, EMPTY, P2, EMPTY, P2, EMPTY, P2, EMPTY],
        [EMPTY, P2, EMPTY, P2, EMPTY, P2, EMPTY, P2]
    ])
    moving_draught = None
    moving_draught_pos = None
    USER_TURN = True

def move (old_pos, new_pos, verify=False):
    """ 
    Method to manage a movement and all the involved stuff. 
    The movement carried out is already supposed to be verified.
    """
    global grid, grid_colors
    if verify and not draughts.verify_move(grid, grid_colors, old_pos, new_pos):
        raise Exception(f"Unpossible movement from {old_pos} to {new_pos} selected by the model")
    grid[new_pos] = grid[old_pos]
    grid[old_pos] = EMPTY
    if manhattan_distance(old_pos, new_pos) == 4:
        mid_pos = (old_pos[0] + (new_pos[0] - old_pos[0]) // 2, old_pos[1] + (new_pos[1] - old_pos[1]) // 2)
        grid[mid_pos] = EMPTY
    # verify if the game is concluded 
    if (grid[grid == P1].size == 0 and grid[grid == P1K].size == 0) or \
       (grid[grid == P2].size == 0 and grid[grid == P2K].size == 0):
        pygame.quit()
        sys.exit()


def user_turn ():
    global USER_TURN, grid, grid_colors, moving_draught, moving_draught_pos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = event.pos
                cell = (int(mouse_y / CELL_HEIGHT), int(mouse_x / CELL_WIDTH))
                if grid[cell] == P2 or grid[cell] == P2K:
                    moving_draught = cell
                    moving_draught_pos = (mouse_x, mouse_y)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and moving_draught is not None:
                mouse_x, mouse_y = event.pos
                cell = (int(mouse_y / CELL_HEIGHT), int(mouse_x / CELL_WIDTH))
                if draughts.verify_move(grid, grid_colors, moving_draught, cell):
                    move(old_pos=moving_draught, new_pos=cell)
                    USER_TURN = False 
                
                moving_draught = None
                moving_draught_pos = None 

        elif event.type == pygame.MOUSEMOTION:
            if moving_draught is not None:
                mouse_x, mouse_y = event.pos
                moving_draught_pos = (mouse_x, mouse_y)


def ai_turn ():
    global grid, grid_colors, agent, TRAIN
    action = agent.move(grid.reshape(1, 8, 8))[-1]
    old_pos, new_pos = draughts.translate_ai_move(grid, grid_colors, action, verify=True)
    move(old_pos, new_pos)


def draw ():
    global USER_TURN, grid, grid_colors, pygame, screen, clock, moving_draught, moving_draught_pos
    grid[0, :] = np.where(grid[0, :] == P2, P2K, grid[0, :])
    grid[7, :] = np.where(grid[7, :] == P1, P1K, grid[7, :])
    screen.fill(BLACK)
    for i, j in itertools.product(range(8), repeat=2):
        if np.array_equal(grid_colors[i, j], WHITE):
            pygame.draw.rect(
                surface=screen, 
                color=grid_colors[i, j], 
                rect=(j * CELL_WIDTH, i * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            )
        
        if grid[i, j] != 0:
            rect = (j * CELL_WIDTH, i * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT) \
                if moving_draught is None or (i, j) != moving_draught \
                else \
                (moving_draught_pos[0] - CELL_WIDTH / 2, moving_draught_pos[1] - CELL_HEIGHT / 2, CELL_WIDTH, CELL_HEIGHT)

            pygame.draw.ellipse(
                surface=screen,
                color=GREY1 if grid[i, j] == P1 or grid[i, j] == P1K else BLUE,
                rect=rect
            )
            if grid[i, j] in (P1K, P2K):
                screen.blit(crown_img, (rect[0] + CELL_WIDTH / 4, rect[1] + CELL_HEIGHT / 4))

    pygame.display.flip()
    clock.tick(FPS)



if __name__ == "__main__":
    while True:
        # User turn
        USER_TURN = True
        while USER_TURN:
            user_turn()
            draw()
        # AI turn
        ai_turn()
        draw()
