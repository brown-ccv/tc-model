from tc_model.model import DDPMUNet_model

def test_model():
    model = DDPMUNet_model()
    assert isinstance(model, object)