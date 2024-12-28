from src.utils.model_loader import get_model_hf

def test_load_model():
	"""Test if model gets loaded from HF"""
	fun_args = {'model_checkpoint': 'nvidia/mit-b0', 'save_path': ''}
	model = get_model_hf(**fun_args)
	assert model is None, "Model was not loaded"
