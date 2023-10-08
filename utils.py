BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY1 = (200, 200, 200)
GREY2 = (150, 150, 150)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)

EMPTY, E = 0, 0
P1, P2 = 1, 2 
P1K, P2K = 3, 4 
 

import math 


def manhattan_distance (pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance (pos1, pos2):
    return math.sqrt(math.pow(pos1[0] - pos2[0], 2) + math.pow(pos1[1] - pos2[1], 2))