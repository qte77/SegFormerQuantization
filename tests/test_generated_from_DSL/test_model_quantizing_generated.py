from src.utils.quantization import quantize_models

def test_quantize_model():
	""""""
	fun_args = {'base_model': ''}
	model, quantized_size, original_size = quantize_models(**fun_args)
	assert model.is_quantized, "Model was not quantized"
	assert quantized_size < original_size, "Quantization did not reduce model size"
