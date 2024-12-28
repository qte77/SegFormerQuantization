from src.utils.model_loader import save_model_hf

def test_save_model():
	""""""
	fun_args = {'save_path': './runtime/models'}
	model = save_model_hf(**fun_args)
	assert model is not None, "Model was loaded"
