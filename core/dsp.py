import numpy as np
from scipy import signal, fft
from scipy.signal import welch


def calculate_resonance_frequency(mass, stiffness):
    """Calculate the resonance frequency for a single degree of freedom system."""
    return np.sqrt(stiffness / mass) / (2 * np.pi)


def simulate_random_response(self, damping_ratio=0.05, force_amplitude=1):
    """Simulate the response of a system to random excitation."""
    natural_freq = dsp.calculate_resonance_frequency(mass=1, stiffness=1)
    response = force_amplitude * np.sin(2 * np.pi * natural_freq * self.time) * np.exp(-damping_ratio * self.time)
    noise = self.generate_white_noise()
    self.signal = response + noise
    return self.signal


def calculate_power_spectral_density(signal, sample_rate):
    """Calculate the Power Spectral Density (PSD) of the signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    frequencies, psd = welch(signal, sample_rate)
    return frequencies, psd


def calculate_fft(self):
    """Calculate the Fast Fourier Transform (FFT) of the signal."""
    if self.signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    fft_values = fft.fft(self.signal)
    fft_freqs = fft.fftfreq(len(self.signal), 1 / self.sample_rate)
    return fft_freqs, fft_values


def calculate_ifft(self, fft_values):
    """Calculate the Inverse Fast Fourier Transform (IFFT) of the FFT values."""
    signal = fft.ifft(fft_values)
    return signal


