import numpy as np

def normalize_input(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return 2 * (data - min_val) / (max(max_val - min_val, 1e-3)) - 1
