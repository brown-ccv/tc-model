import tensorflow as tf
import numpy as np
from pathlib import Path
from tc_model.model import DDPMUNet_model, run_model, make_prediction
from tc_model.weighted_inputs import load_weighted_inputs

def test_unet_model():
    model = DDPMUNet_model()
    assert isinstance(model, object)

def test_run_model():
    project_root = Path(__file__).parents[1]
    tensor = run_model(load_weighted_inputs([1, 0, 0, 0], project_root / "src" / "tc_model" / "data"))
    assert isinstance(tensor, tf.Tensor)
    shape = tensor.shape
    assert isinstance(shape, tf.TensorShape)
    assert shape[0] == 110 
    assert shape[1] == 210
    assert shape[2] == 1 
    assert tensor.dtype == tf.float32
    assert isinstance(tensor.numpy(), np.ndarray)

def test_run(): 
    prediction = make_prediction([1, 0, 0, 0])
    assert isinstance(prediction, list)
    assert len(prediction) == 110
    assert isinstance(prediction[0], list)
    assert len(prediction[0]) == 210
    assert isinstance(prediction[0][0], list)
    assert len(prediction[0][0]) == 1
    assert isinstance(prediction[0][0][0], float)
    