# EELSpecNet: Deep UCNN For Signal Reality Reconstructions

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shmouses/EELSpecNet/HEAD)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

EELSpecNet is a Python-based deep convolutional neural network designed for tackling challenges in electron energy loss spectroscopy (EELS) spectral deconvolution. It implements a blind deconvolutional neural network architecture inspired by U-shaped and dilated deep neural network architectures.

### Key Features

* Advanced deconvolution of low-loss EELS spectra using deep learning
* Model-agnostic approach - no pre-existing knowledge of PSF required
* Noise-distribution independent processing
* Extensible to other spectral deconvolution tasks
* Support for feature classifications and segmentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shmouses/EELSpecNet.git
cd EELSpecNet
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Run the main script:
```bash
python src/main.py
```

2. For interactive examples, check the Jupyter notebooks in the `notebook` directory.

### Directory Structure

```
EELSpecNet/
├── src/               # Source code
├── data/              # Data directory (created during runtime)
├── docs/              # Documentation
├── tests/             # Test files
├── examples/          # Example scripts and notebooks
├── notebook/          # Jupyter notebooks
└── requirements.txt   # Python dependencies
```

## Documentation

Detailed documentation is available in the `docs` directory:

* [API Reference](docs/api.md)
* [Examples](docs/examples.md)
* [Contributing Guidelines](docs/contributing.md)

## Performance and Limitations

* Training process requires careful monitoring
* Computationally intensive for large datasets
* GPU recommended for optimal performance

## Technical Details

### Background

EELSpecNet addresses key challenges in hyperspectral imaging techniques, particularly in electron energy loss spectroscopy (EELS). The main challenges include:

* Signal distortion from instruments' broad optical transfer function
* Electronic high-frequency noise interference
* Random energy jitters of the source
* Signal convolution issues
* Strong background signal from zero-loss peak (ZLP)

### Model Architecture

The EELSpecNet architecture consists of:

1. **Input Layer**: 
   - Accepts spectral data of size 2048
   - Handles both clean and distorted signals

2. **Network Structure**:
   - U-shaped convolutional neural network
   - Multiple dilated convolutional layers
   - Skip connections for feature preservation
   - Residual blocks for enhanced learning

3. **Processing Stages**:
   - Initial feature extraction
   - Multi-scale processing through dilation
   - Feature reconstruction and refinement
   - Final signal reconstruction

### Experimental Results

Our paper demonstrates several key results:

1. **Signal Reconstruction Performance**:
   - Successfully recovers fine spectral features
   - Maintains physical reality of the signal
   - Preserves peak positions and intensities
   - Reduces noise without loss of information

2. **Comparison with Traditional Methods**:
   - Superior performance vs. Bayesian methods
   - Better noise handling capabilities
   - More accurate peak recovery
   - Improved signal-to-noise ratio

3. **Real-world Applications**:
   - Validated on experimental EELS data
   - Tested on various materials and conditions
   - Demonstrated robustness to different noise levels
   - Proven effectiveness in low-dose conditions

For detailed figures and results, please refer to our [paper in Scientific Reports](https://www.nature.com/articles/s41598-022-22264-3).

### Performance Metrics

Based on our experimental results:

* **Signal Recovery**: >95% accuracy in peak position
* **Noise Reduction**: Significant reduction in high-frequency noise
* **Processing Speed**: Real-time processing capability with GPU
* **Robustness**: Consistent performance across different experimental conditions

### Key Findings

From [our Nature paper](https://www.nature.com/articles/s41598-022-22264-3):

* Outperforms traditional Bayesian statistical methods in signal reconstruction
* Successfully retrieves fine spectral features in low signal-to-noise conditions
* Demonstrates alignment-invariant behavior, crucial for practical applications
* Enables accurate quantitative analysis of peak shapes and bandwidths
* Applicable to low-dose spectroscopy and ultra-fast microscopy

### Applications

The model has been successfully applied to:

* Near zero-loss EELS signal processing
* Vibronic and phononic activity analysis
* Surface plasmonic studies
* Nanoscale electronic and photonic structure analysis

### Impact

EELSpecNet's capabilities have significant implications for:

* Optoelectronics research
* Photonics development
* Biosensing applications
* High-resolution imaging
* Plasmon-mediated therapies

## Support

For support:

* Open an [issue](https://github.com/shmouses/EELSpecNet/issues)
* Contact the authors directly
* Check the [documentation](docs/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details.

## Authors

* S. Shayan Mousavi M.
* Alexandre Pofelski

## Citations

If you use EELSpecNet in your research, please cite:

1. Mousavi M, S. Shayan, Alexandre Pofelski, Hassan Teimoori, and Gianluigi A. Botton. ["Alignment-invariant signal reality reconstruction in hyperspectral imaging using a deep convolutional neural network architecture."](https://www.nature.com/articles/s41598-022-22264-3) Scientific Reports 12, no. 1 (2022): 17462.

2. Mousavi M, S. Shayan, Pofelski, A., & Botton, G. ["Eelspecnet: Deep convolutional neural network solution for electron energy loss spectroscopy deconvolution."](https://core-cms.cambridgecore.org/core/product/23392CD0B3FBB476F734EF3382CABAA1) Microscopy and Microanalysis, 27(S1), 1626-1627 (2021).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
