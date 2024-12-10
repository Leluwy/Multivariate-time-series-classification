# Multivariate Time-Series Classification with Shape-Based Interpretability

### 1. **`samples_analyze.py`**
This file is responsible for creating visualizations to analyze multivariate data. Key functionalities include:
- Plotting the original multivariate input samples.
- Generating visualizations for the class activation maps.
- Producing plots for filter (convolution) visualization.

### 2. **Neural Networks**
This section contains scripts and functions related to the neural network classifiers.

#### Files:
- **`main.py`**: Contains the main settings and logic for running the code.
- **`utils.py`**: Provides supporting functions such as reading input data. It also includes functions for interpreting deep learning methods, including:
  - Multidimensional Scaling (MDS)
  - Occlusion Method
  - Filter Visualization

#### Neural Network Models:
- **`cnn.py`**: Implements Convolutional Neural Network.
- **`encoder.py`**: Implements Encoder architectures.
- **`fcn.py`**: Implements Fully Convolutional Network.
- **`mlp.py`**: Implements Multilayer Perceptron.
- **`resnet.py`**: Implements Residual Network.

### 3. **`PSD_and_Scattering.py`**
This file contains all the experiments conducted with Random Forest classifiers using Power Spectral Density (PSD) and wavelet scattering features.

### 4. **`wavelets.py`**
This file contains experiments using Random Forest classifiers on wavelet features.

### 5. **`plot_wavelets.py`**
This script is used to generate plots of wavelet functions.

### 6. **`brain_overlay.py`**
This script generates brain images that visualize the relevance of electrodes in the dataset.

