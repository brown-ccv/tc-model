import os
import numpy as np
from pathlib import Path

basin = "SP"
model_names = ["CMCC-CM2-VHR4", "EC-Earth3P-HR", "CNRM-CM6-1-HR", "HadGEM3-GC31-HM"]
data_files = ["GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy".format(model) for model in model_names]

def get_gcm_genesis_maps(data_folder: Path):
    """
    Load the gcm genesis maps

    @param genesis_data_folder: path to the folder containing the gcm genesis maps
    @return: array of the genesis maps generated from the gcm data
    """

    genesis_files = [data_folder / data_file for data_file in data_files]

    future_data = [
        np.load(file_path, allow_pickle=True).item()[basin]
        for file_path in genesis_files
    ]

    return future_data

def get_genesis_map_from_weights(weights: list, gcm_maps: list):
    """
    @param weights: a list of 4 floats that sum to 1 representing a linear weighting between the gcm maps.
    @param gcm_maps: gcm maps from get_gcm_genesis_maps
    @return:
    """
    if len(weights) != 4 or sum(weights) != 1:
        raise Exception()

    monthlist =  [1, 2, 3, 4, 11, 12]

    genesis_location_matrices = {}

    for month in monthlist:
        genesis_data_for_month = [
            np.nan_to_num(gcm_maps[i][month]) for i in range(4)
        ]

        grids = np.array(genesis_data_for_month)
        randomized_grid = (grids.T @ np.array(weights)).T

        genesis_location_matrices[month] = randomized_grid

    return genesis_location_matrices

def load_weighted_inputs(weights: list, data_folder: Path):
    genesis_maps = get_gcm_genesis_maps(data_folder)
    return get_genesis_map_from_weights(weights, genesis_maps)