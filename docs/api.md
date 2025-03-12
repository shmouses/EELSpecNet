# API Reference

## EELSpecNet Model

### `EELSpecNetModel_CNN_10D`

The main model class for spectral deconvolution.

#### Parameters

- `input_size` (int): Size of the input spectrum (default: 2048)

#### Methods

- `compile(optimizer, loss, metrics)`: Compile the model with specified optimizer and loss function
- `fit(x, y, validation_split, batch_size, epochs)`: Train the model on input data
- `predict(x)`: Generate predictions for input data
- `evaluate(x, y)`: Evaluate model performance on test data

## Data Generation

### `GenerateData`

Module for generating training and evaluation data.

#### Functions

- `training_signal_set(size, snr, psf_width_min, psf_width_max, dim, noise_level)`: Generate training data
- `eval_signal_set(size, snr, psf_width_min, psf_width_max, dim, noise_level)`: Generate evaluation data

## Usage Examples

```python
import EELSpecNet
import GenerateData as gene

# Create model
model = EELSpecNet.EELSpecNetModel_CNN_10D(2048)

# Generate training data
train_target, train_initial = gene.training_signal_set(
    6000,  # size
    -2,    # snr
    0.005, # psf_width_min
    0.015, # psf_width_max
    2048,  # dim
    0.05   # noise_level
)

# Train model
model.fit(train_initial, train_target, 
         validation_split=0.16,
         batch_size=16,
         epochs=1000)
``` 