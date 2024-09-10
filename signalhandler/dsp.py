import numpy as np
from scipy import fft
from scipy.signal import welch, butter, filtfilt

import core.viz


def normalize_signal(signal):
    """Normalize the signal to have zero mean and unit variance."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal

def calculate_resonance_frequency(mass, stiffness):
    """Calculate the resonance frequency for a single degree of freedom system."""
    return np.sqrt(stiffness / mass) / (2 * np.pi)


def simulate_random_response(self, damping_ratio=0.05, force_amplitude=1):
    """Simulate the response of a system to random excitation."""
    natural_freq = calculate_resonance_frequency(mass=1, stiffness=1)
    response = force_amplitude * np.sin(2 * np.pi * natural_freq * self.time) * np.exp(-damping_ratio * self.time)
    noise = self.generate_white_noise()
    self.signal = response + noise
    return self.signal




def calculate_fft(signal, sample_rate):
    """Calculate the Fast Fourier Transform (FFT) of the signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    fft_values = fft.fft(signal)
    fft_freqs = fft.fftfreq(len(signal), 1 / sample_rate)
    return fft_freqs, fft_values


def _apply_filter(signal, sample_rate, cutoff_frequency, filter_type, order):
    """Helper method to apply filters."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a, _ = butter(order, normal_cutoff, btype=filter_type, output='ba', fs=sample_rate)
    signal = filtfilt(b, a, signal)
    return signal
