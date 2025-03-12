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

1. Mousavi M, S. Shayan, Alexandre Pofelski, Hassan Teimoori, and Gianluigi A. Botton. "Alignment-invariant signal reality reconstruction in hyperspectral imaging using a deep convolutional neural network architecture." Scientific Reports 12, no. 1 (2022): 17462.

2. Mousavi M, S. Shayan, Pofelski, A., & Botton, G. (2021). Eelspecnet: Deep convolutional neural network solution for electron energy loss spectroscopy deconvolution. Microscopy and Microanalysis, 27(S1), 1626-1627.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
