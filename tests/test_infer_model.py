import torch
from src.utils.evaluator import infer_model
from unittest.mock import  MagicMock

class MockModel:
    def __call__(self, pixel_values, labels):
        return MagicMock(loss=0.5, logits=torch.tensor([1.0]))

def test_infer_model():
    model = MockModel()
    pixel_values = torch.tensor([1.0])
    labels = torch.tensor([1.0])
    loss, logits = infer_model(model, pixel_values, labels)
    assert loss == 0.5
    assert torch.equal(logits, torch.tensor([1.0]))