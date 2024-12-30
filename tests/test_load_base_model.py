from unittest.mock import patch
from src.utils.model_loader import load_base_model

@patch("src.utils.model_loader.SegformerForSemanticSegmentation.from_pretrained")
@patch("src.utils.model_loader.exists")
@patch("src.utils.model_loader.makedirs")
def test_load_base_model(mock_makedirs, mock_exists, mock_from_pretrained):
    mock_exists.return_value = False
    mock_from_pretrained.return_value = "model"
    model = load_base_model("model_name", "model_save_path", "dtype", "device")
    assert model == "model"
    mock_from_pretrained.assert_called()
    mock_makedirs.assert_called_once_with("model_save_path", exist_ok=True)