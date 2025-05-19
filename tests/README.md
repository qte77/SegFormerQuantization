# Refactor/Rewrite for TDD first approach

## Approach

1. Identify Core Functionality
   - Prioritize key functionality first
2. Write Tests First
   - Test functionality, not implementation
3. Refactor Implementation
   - Red/Green/Blue

## First functions

Start with following base functions in `utils/`:

1. `test_model_loading`: `load_base_model` in `model_loader.py`
2. `test_dataset`: `load_dataset_custom` in `data_processing.py`
3. `test_predict`: `infer_model` in `evaluator.py`
4. `test_wandb`: `create_wandb` in `wandb_utils.py`
5. `test_utils`: `print_model_sizes` in `general_utils.py`
6. optional: `test_end_to_end`
