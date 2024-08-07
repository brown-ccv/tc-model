import numpy.testing as t
from tc_model.model import model

def test_model():
    t.assert_equal(model(), 1)