Project title: Multivariate time-series classification with shape-base interpretability

This file will provide a brief overview of the various functions used in this project

1.) samples_analyze.py 

This file primarily serves to create plots for analyzing multivariate data. It encompasses tasks such as plotting the original (multivariate) input sample, generating visualizations for the class activation map, and producing plots for filter (convolution) visualization.

2.) Neural Networks

This code is used for the neural network classifiers.

-main.py contains the major setting for running the code
-utils.py contains all the supporting functions (e.g. read input data). Furthermore, it incorporates functions for interpreting deep learning methods, such as MDS, Occlusion method, and Filter Visualization.

Also, it includes the function classes for the deep learning methods:
- cnn.py 
-encoder.py
-fcn.py 
-mlp.py
-resnet.py 

3.) PSD_and_Scattering.py

This file contains all the Random Forest experiments conducted on the PSD and wavelet scattering features. 

4.) wavelets.py

This file contains all the Random Forest experiments conducted on the wavelet features. 

5.) plot_wavelets.py

This function is used to plot the wavelet functions.

6.) brain_overleay.py

The code within this file includes the function utilized to generate a brain image that visualizes the relevance of electrodes.

