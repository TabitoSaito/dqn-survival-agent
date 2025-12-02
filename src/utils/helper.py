import random
import numpy as np

def get_unique_coordinates(shape: tuple[int, int], n: int):
    row, col = shape
    return [np.array([value // row, value % col], dtype=int) for value in random.sample(range(row * col), n)]