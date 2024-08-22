import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
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


def _apply_filter(signal, sample_rate, cutoff_frequency, filter_type, order):
    """Helper method to apply filters."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, output='ba', fs=sample_rate)

    signal = filtfilt(b, a, signal)
    return signal

def process_in_chunks(signal=None, chunk_size=None):
    """Process the signal in chunks for real-time applications.
    :param chunk_size:
    :param signal:
    """
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    num_chunks = len(signal) // chunk_size
    for i in range(num_chunks):
        chunk = signal[i*chunk_size:(i+1)*chunk_size]
        yield chunk



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