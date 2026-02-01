# MLflow Run Analysis Summary

## Run Information
- **Run ID**: 05031104766646aa8e03a3cac5aa48ae
- **Status**: FINISHED
- **Duration**: 21.8 minutes
- **Date**: February 1, 2026

## Configuration
- **Model**: EfficientNet-B3 + Trimodal Fusion
- **Epochs**: 5
- **Batch Size**: 8
- **Image Size**: 224x224
- **Learning Rate**: 2e-4 → 1e-4
- **Dropout**: 0.5

## Performance Results

### Training Metrics
- **Accuracy**: 97.13%
- **Loss**: 0.554
- **F1-Macro**: 97.14%
- **F1 per class**:
  - 1_person: 97.05%
  - Bicycle: 97.20%
  - Car: 97.15%

### Validation Metrics
- **Accuracy**: 38.12% ⚠️
- **Loss**: 1.292
- **F1-Macro**: 31.70%
- **F1 per class**:
  - 1_person: 32.54%
  - Bicycle: 12.58%
  - Car: 50.00%

## Critical Issues
1. **Severe Overfitting**: 59% train-val accuracy gap
2. **Poor Generalization**: Validation performance near random
3. **Class Imbalance**: Bicycle class severely underperforming

## Recommendations
1. Increase regularization (dropout 0.5→0.7)
2. Enhance data augmentation
3. Reduce model complexity
4. Extend training with early stopping
5. Investigate feature engineering

## Files Generated
- `trimodal_training_report.tex` - Complete Overleaf LaTeX report
- `mlflow_run_data.json` - Full MLflow metrics export