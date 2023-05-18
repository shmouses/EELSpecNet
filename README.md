# EELSpecNet: A Deep UCNN For Signal Reality Reconstructions  

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shmouses/EELSpecNet/HEAD)

## Description
EELSpecNet is a Python-based deep convolutional neural network designed for tackling challenges in electron energy loss spectroscopy (EELS) spectral deconvolution. EELS is a powerful technique for studying the chemical and electronic properties of materials at the nanometer length-scale. It is capable of performing near-meV energy-resolution spectroscopy, exploring plasmonic and phononic activities, and measuring energy gains. However, the output spectra often suffer from high-frequency noise and convolution with the optical transfer function (OTF). EELSpecNet offers a solution to this problem by implementing a blind deconvolutional neural network architecture inspired by the U-shaped and dilated deep neural network architectures. 

## Key Features
* Deconvolves low-loss EELS spectra using deep learning.
* Not dependent on pre-existing knowledge (like the PSF) and assumptions on the noise distribution.
* Capable of extending to other spectral deconvolution (not limited to EELS), feature classifications, and segmentation with minor modifications and a dedicated training set.

## Limitations
The training process needs to be closely monitored and can be computationally expensive.

## Requirements
EELSpecNet is a Python script. The specific Python version and libraries required will be specified in the 'Installation' section. 

## Installation
(Provide specific instructions about how to install the software including required Python version and libraries)

## Usage
(Provide instructions on how to use the software. This can include command-line examples, function calls in Python, etc.)

## Support
For any issues, bugs, feature requests, or questions about EELSpecNet, please open an issue in the issue tracker, or contact the authors directly.

## Authors
* S. Shayan Mousavi M.
* Alexandre Pofelski

## References
(Provide a list of references here, as cited in the software's documentation or in the provided description)

## License
(Include information about the software license here)

## How to Cite
1. Mousavi M, S. Shayan, Alexandre Pofelski, Hassan Teimoori, and Gianluigi A. Botton. "Alignment-invariant signal reality reconstruction in hyperspectral imaging using a deep convolutional neural network architecture." Scientific Reports 12, no. 1 (2022): 17462.
2. Mousavi M, S. Shayan, Pofelski, A., & Botton, G. (2021). Eelspecnet: Deep convolutional neural network solution for electron energy loss spectroscopy deconvolution. Microscopy and Microanalysis, 27(S1), 1626-1627.
