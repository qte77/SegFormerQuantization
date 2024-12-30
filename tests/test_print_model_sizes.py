from src.utils.general_utils import print_model_sizes
from unittest.mock import MagicMock

def test_print_model_sizes(capfd):
    model_mock = MagicMock()
    model_mock.get_memory_footprint.return_value = 1e6
    model_mock.device = "cpu"
    models = {"model1": model_mock}
    print_model_sizes(models)
    captured = capfd.readouterr()
    assert "model1 size 1.00 MB on cpu" in captured.out