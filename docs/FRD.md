# Functional Requirements Document (FRD)

## System Components

### 1. Model Management

- **Load pre-trained SegFormer models** from local storage or Hugging Face.
- **Apply quantization techniques** (float8, int8, int4, int2) to reduce model size and potentially improve inference speed.
- **Save and load quantized models** for future use or comparison.

### 2. Data Processing

- **Load and preprocess the Scene Parse 150 dataset** for semantic segmentation tasks.
- **Implement efficient data sharding** to manage large datasets, allowing for batch processing.
- **Convert images and annotations** into the required format for model input.

### 3. Evaluation Pipeline

- **Implement inference function** for SegFormer models to generate predictions.
- **Calculate evaluation metrics** (mean IoU, mean accuracy, overall accuracy) to assess model performance.
- **Process evaluation in batches** for efficiency, ensuring the system can handle large datasets.

### 4. Experiment Tracking

- **Initialize and manage Weights & Biases runs** for logging and visualization of experiments.
- **Log model performance metrics and metadata** to track changes over time.
- **Create visualizations** for easy comparison of different quantization methods.

### 5. Utility Functions

- **Manage environment variables and configuration** through `src/config.py`.
- **Implement helper functions** for data conversion, file management, and error handling.

## Functional Requirements

1. **Model Loading and Quantization**
   - The system shall load pre-trained SegFormer models from local storage or Hugging Face.
   - The system shall apply specified quantization techniques to the loaded models.

2. **Dataset Processing**
   - The system shall efficiently load and preprocess the Scene Parse 150 dataset.
   - The system shall implement a data sharding mechanism to handle large datasets.

3. **Evaluation Pipeline**
   - The system shall perform inference using the quantized models on the preprocessed data.
   - The system shall calculate and log evaluation metrics for each quantized model.

4. **Experiment Tracking**
   - The system shall integrate with Weights & Biases for experiment tracking and visualization.

5. **Modular Design**
   - The system shall provide a modular structure allowing easy extension to additional models and datasets.

## Performance Requirements

1. **Evaluation Speed**
   - The evaluation pipeline shall process the entire Scene Parse 150 validation set in under 24 hours on a single GPU.

2. **GPU Utilization**
   - The system shall use GPU acceleration when available to speed up model inference.

3. **Memory Usage**
   - The memory usage shall not exceed 12GB of GPU memory at any point during execution.

## Usability Requirements

1. **Console Output**
   - The system shall provide clear console output indicating progress and any errors.

2. **Configuration**
   - Configuration of experiments shall be possible through a single configuration file (`src/config.py`).

3. **Logging and Visualization**
   - The system shall generate comprehensive logs and visualizations in Weights & Biases for easy analysis.

