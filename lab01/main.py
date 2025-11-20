import numpy as np

FUNCTION_CONST = 1
GRID_SIZE = 0.001
GRID_COUNT = 1000

right_hand_size = - 1 * FUNCTION_CONST * GRID_SIZE

m = np.zeros((GRID_COUNT, GRID_COUNT))
m[1:GRID_COUNT-1, 1:GRID_COUNT-1] = 1
m = m.flatten()
m_new = np.zeros_like(m)

def gauss_seidel(index_arr):
    pass