from unittest.mock import patch
from src.utils.wandb_utils import create_wandb_run

@patch("src.utils.wandb_utils.init")
def test_create_wandb_run(mock_init):
    mock_init.return_value = "wandb_run"
    result = create_wandb_run("project_name", "entity_name", "run_name", "group_name")
    assert result == "wandb_run"
    mock_init.assert_called_once_with(project="project_name", entity="entity_name", name="run_name", group="group_name")