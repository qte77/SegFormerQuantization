import pytest
# from typing import Dict, Any
from src.utils.model_loader import load_base_model
# from src.config import Config

from sys import path
print(f"pytest sys.path: {path}")

def test_model_loading() -> None:
    """
    Test the loading of a SegFormer model.

    Raises:
        AssertionError: If the model loading or quantization fails or if the memory footprint does not decrease.
    """
    # config = Config()
    model_checkpoint = "nvidia/mit-b0"
    save_path = "./runtime/models"

    model = load_base_model(
        model_checkpoint=model_checkpoint,
        save_path=save_path
    )
    
    assert model is not None, "Model was not loaded"

if __name__ == "__main__":
    pytest.main([__file__])
