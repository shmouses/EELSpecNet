# Examples

This document provides examples of how to use EELSpecNet for various tasks.

## Basic Usage

### Training a Model

```python
import EELSpecNet
import GenerateData as gene
import numpy as np

# Create and compile model
model = EELSpecNet.EELSpecNetModel_CNN_10D(2048)
model.compile(
    optimizer='adam',
    loss='BinaryCrossentropy',
    metrics=['mape', 'mse']
)

# Generate training data
train_target, train_initial = gene.training_signal_set(
    size=6000,
    snr=-2,
    psf_width_min=0.005,
    psf_width_max=0.015,
    dim=2048,
    noise_level=0.05
)

# Train model
model.fit(
    train_initial,
    train_target,
    validation_split=0.16,
    batch_size=16,
    epochs=1000
)
```

### Evaluating Model Performance

```python
# Generate evaluation data
eval_target, eval_initial, eval_peaks, eval_psf, eval_metadata = gene.eval_signal_set(
    size=2000,
    snr=-2,
    psf_width_min=0.005,
    psf_width_max=0.015,
    dim=2048,
    noise_level=0.05
)

# Evaluate model
model.evaluate(eval_initial, eval_target)

# Make predictions
predictions = model.predict(eval_initial)
```

## Advanced Usage

### Custom Data Processing

```python
# Reshape data for model input
x_dim, e_dim = np.shape(train_initial)
train_initial = train_initial.reshape((x_dim, 1, e_dim, 1))
train_target = train_target.reshape((x_dim, 1, e_dim, 1))

# Add small offset to prevent numerical issues
train_initial += 0.001
train_target += 0.001
```

### Saving Predictions

```python
# Save predictions to file
predictions = model.predict(eval_initial)
predictions = predictions.reshape((2000, 2048))
np.save("deconv_evaluation_signal.npy", predictions)
```

## Tips and Best Practices

1. Always normalize your input data
2. Use appropriate batch sizes based on your GPU memory
3. Monitor training progress to prevent overfitting
4. Save model checkpoints during training
5. Use validation split to assess model performance 