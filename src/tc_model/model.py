import numpy as np
import os
from pathlib import Path

from tc_model import ddpm_unet
from tc_model.utils.data import normalize_input

project_root = Path(__file__).parents[1]

class DDPMUNet_model:
    
    MODEL_CONFIG = {
        "img_size": (112, 224, 1),
        "output_size": (110, 210, 1),
        "has_attention": [False, False, True],
        "interpolation": "bilinear",
        "widths": [16, 32, 64],
        "include_temb": False
    }
    
    model_path = project_root / "tc_model" / "weights.keras" #os.path.join(os.path.curdir, "src/tc_model/weights.keras") #os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), "weights.keras")
 
    def __init__(self):
        self.model = DDPMUNet_model.load_model()

    @staticmethod
    def load_model():
        #model = keras.saving.load_model(DDPMUNet_model.model_path)
        model = ddpm_unet.build_model(**DDPMUNet_model.MODEL_CONFIG)
        model.load_weights(DDPMUNet_model.model_path)
        return model

    @staticmethod
    def preprocess_input(genesis: np.ndarray):
        """
        @param genesis: (months = 6, lat = 55, lon = 105) shaped np.ndarray
        @return: (lat = 110, lon=210, channels = 1) shaped np.ndarray
        with values normalized [-1, 1]
        """

        month_axis = 0
        genesis_month_sum = np.sum(genesis, axis=month_axis)

        # this is a simple way to do a nearest neighbor upsample
        scaling_factor = 2
        scaling_matrix = np.ones((scaling_factor, scaling_factor))
        upsampled_genesis = np.kron(genesis_month_sum, scaling_matrix)

        # we pad the inputs so that each dimension is divisible by 8
        # upsampled_genesis has shape (110, 210)
        # the closest shape with dimensions divisible by 8 is (112, 224)
        lat_padding = (1, 1)
        lon_padding = (7, 7)

        padded_genesis = np.pad(upsampled_genesis, (lat_padding, lon_padding))

        # normalize and add channel dimension
        normalized_genesis = normalize_input(np.expand_dims(padded_genesis, axis=-1))
        return normalized_genesis

    def __call__(self, x):
        x = self.preprocess_input(x)
        x = x[np.newaxis, :]
        return self.model(x)[0]
