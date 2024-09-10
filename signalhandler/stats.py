import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

def calculate_expectation(signal):
    """Calculate the expected value of the signal."""
    return np.mean(signal)


def calculate_covariance(signal, other_signal):
    """Calculate the covariance of the signal with another signal."""
    if signal is None or other_signal is None:
        raise ValueError("Both signals must be generated first.")
    return np.cov(signal, other_signal)[0, 1]
def calculate_autocorrelation(signal, time):
    """Calculate the auto-correlation of the signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    return time[:autocorr.size], autocorr


def calculate_cross_correlation(signal, other_signal, sample_rate):
    """Calculate the cross-correlation of the signal with another signal."""
    if signal is None or other_signal is None:
        raise ValueError("Both signals must be generated first.")
    crosscorr = np.correlate(signal, other_signal, mode='full')
    lags = np.arange(-len(signal) + 1, len(signal))
    return lags / sample_rate, crosscorr


def generate_normal_distribution(time, mean=0, std=1):
    """Generate a normally distributed signal."""
    sinal = np.random.normal(mean, std, len(time))
    return sinal


def calculate_skewness(signal):
    """Calculate the skewness of the signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    return skew(signal)


def calculate_kurtosis(signal):
    """Calculate the kurtosis of the signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    return kurtosis(signal)








def cum_dens_func(signal=None, plot=True):
    """Plot the Cumulative Distribution Function (CDF) of the signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    ecdf = np.sort(signal)
    cdf = np.arange(1, len(signal) + 1) / len(signal)
    if plot:
        plt.plot(ecdf, cdf)
        plt.title("Cumulative Distribution Function (CDF)")
        plt.xlabel("Signal Amplitude")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.show()
    else:
        return cdf, ecdf