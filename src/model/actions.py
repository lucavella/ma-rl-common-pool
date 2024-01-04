import numpy as np



ORIENT_NORTH = 0
ORIENT_EAST  = 1
ORIENT_SOUTH = 2
ORIENT_WEST  = 3

N_ORIENTATIONS = 4
ORIENTATIONS   = range(N_ORIENTATIONS)

STEP_FORWARD  = 0
STEP_RIGHT    = 1
STEP_BACKWARD = 2
STEP_LEFT     = 3
ROTATE_LEFT   = 4
ROTATE_RIGHT  = 5
TAG_AHEAD     = 6
STAND_STILL   = 7

N_ACTIONS    = 8
STEP_RANGE   = range(4)
ROTATE_RANGE = range(4, 6)


def rotate_orientation(orientation, rotation_direction):
    rotation_term = rotation_direction * 2 - (ROTATE_LEFT + ROTATE_RIGHT) # -1 for left, +1 for right
    return (orientation + rotation_term) % N_ORIENTATIONS


def move_direction(direction, orientation):
    return (direction + orientation) % N_ORIENTATIONS


def rotate_position(position, orientation, shape):
    y, x = position
    h, w = shape
    if orientation == ORIENT_NORTH:
        return y, x
    elif orientation == ORIENT_EAST:
        return w - x - 1, y
    elif orientation == ORIENT_SOUTH:
        return h - y - 1, w - x - 1
    elif orientation == ORIENT_WEST:
        return x, h - y - 1