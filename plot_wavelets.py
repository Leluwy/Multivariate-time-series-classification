# Import packages
import pywt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Wavelet function
    wavelet_function = 'sym2'

    # Plot the wavelet functions
    plt.figure()

    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)

    x = np.linspace(-3, 3, 97)
    wavelet_function = pywt.Wavelet(wavelet_function).wavefun(level=5)[1]
    plt.plot(x, wavelet_function, label=wavelet_function, linewidth=3)

    # Add labels
    plt.xlabel('x', fontsize=24)
    plt.ylabel('\u03C8(x)', fontsize=24)

    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontsize(28)

    # Display the plot
    plt.show()
