import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import csv
import scipy.io
import h5py


class SignalHandler:
    def __init__(self, sample_rate=1000, duration=10):
        """
        Initialize the SignalHandler with a specific sample rate and duration.

        :param sample_rate: Number of samples per second (Hz)
        :param duration: Duration of the signal in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        self.signal = None

    def apply_low_pass_filter(self, cutoff_frequency, order=5):
        """Apply a low-pass filter to the signal."""
        return self._apply_filter(cutoff_frequency, 'low', order)

    def apply_high_pass_filter(self, cutoff_frequency, order=5):
        """Apply a high-pass filter to the signal."""
        return self._apply_filter(cutoff_frequency, 'high', order)

    def apply_band_pass_filter(self, low_cutoff, high_cutoff, order=5):
        """Apply a band-pass filter to the signal."""
        nyquist = 0.5 * self.sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal

    def _apply_filter(self, cutoff_frequency, filter_type, order):
        """Helper method to apply filters."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal

    def process_signal_in_chunks(self, chunk_size):
        """Process the signal in chunks for real-time applications."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        num_chunks = len(self.signal) // chunk_size
        for i in range(num_chunks):
            chunk = self.signal[i*chunk_size:(i+1)*chunk_size]
            yield chunk

# dsp
    def calculate_power_spectral_density(self):
        """Calculate the Power Spectral Density (PSD) of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        frequencies, psd = signal.welch(self.signal, self.sample_rate)
        return frequencies, psd
    
    def calculate_fft(self):
        """Calculate the Fast Fourier Transform (FFT) of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        fft_values = fft.fft(self.signal)
        fft_freqs = fft.fftfreq(len(self.signal), 1/self.sample_rate)
        return fft_freqs, fft_values

    def calculate_ifft(self, fft_values):
        """Calculate the Inverse Fast Fourier Transform (IFFT) of the FFT values."""
        signal = fft.ifft(fft_values)
        return signal

    def apply_moving_average_filter(self, window_size=5):
        """Apply a simple moving average filter to the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        self.signal = np.convolve(self.signal, np.ones(window_size)/window_size, mode='valid')
        return self.signal

# stoch
    def generate_white_noise(self, mean=0, std=1):
        """Generate a white noise signal."""
        self.signal = np.random.normal(mean, std, len(self.time))
        return self.signal

    def generate_random_vibration(self, frequency=5, amplitude=1):
        """Generate a random vibration signal."""
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * self.time)
        noise = self.generate_white_noise()
        self.signal = sine_wave + noise
        return self.signal

    def generate_gaussian_process(self, mean=0, std=1):
        """Generate a Gaussian stochastic process."""
        self.signal = np.random.normal(mean, std, len(self.time))
        return self.signal

    def generate_poisson_process(self, lam=1):
        """Generate a Poisson stochastic process."""
        self.signal = np.random.poisson(lam, len(self.time))
        return self.signal

    def simulate_ornstein_uhlenbeck(self, theta=0.15, mu=0, sigma=0.3):
        """Simulate the Ornstein-Uhlenbeck process."""
        dt = 1/self.sample_rate
        x = np.zeros(len(self.time))
        for i in range(1, len(self.time)):
            x[i] = x[i-1] + theta*(mu - x[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal()
        self.signal = x
        return self.signal

    def calculate_expectation(self):
        """Calculate the expected value of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        return np.mean(self.signal)

    def calculate_covariance(self, other_signal):
        """Calculate the covariance of the signal with another signal."""
        if self.signal is None or other_signal is None:
            raise ValueError("Both signals must be generated first.")
        return np.cov(self.signal, other_signal)[0, 1]

# vibs
    def calculate_resonance_frequency(self, mass, stiffness):
        """Calculate the resonance frequency for a single degree of freedom system."""
        return np.sqrt(stiffness / mass) / (2 * np.pi)

    def simulate_random_response(self, damping_ratio=0.05, force_amplitude=1):
        """Simulate the response of a system to random excitation."""
        natural_freq = self.calculate_resonance_frequency(mass=1, stiffness=1)
        response = force_amplitude * np.sin(2 * np.pi * natural_freq * self.time) * np.exp(-damping_ratio * self.time)
        noise = self.generate_white_noise()
        self.signal = response + noise
        return self.signal

# stats
    def calculate_autocorrelation(self):
        """Calculate the auto-correlation of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        autocorr = np.correlate(self.signal, self.signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        return self.time[:autocorr.size], autocorr

    def calculate_cross_correlation(self, other_signal):
        """Calculate the cross-correlation of the signal with another signal."""
        if self.signal is None or other_signal is None:
            raise ValueError("Both signals must be generated first.")
        crosscorr = np.correlate(self.signal, other_signal, mode='full')
        lags = np.arange(-len(self.signal) + 1, len(self.signal))
        return lags / self.sample_rate, crosscorr
    
    def generate_normal_distribution(self, mean=0, std=1):
        """Generate a normally distributed signal."""
        self.signal = np.random.normal(mean, std, len(self.time))
        return self.signal

    def calculate_skewness(self):
        """Calculate the skewness of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        return stats.skew(self.signal)

    def calculate_kurtosis(self):
        """Calculate the kurtosis of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        return stats.kurtosis(self.signal)

    def plot_cdf(self):
        """Plot the Cumulative Distribution Function (CDF) of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        ecdf = np.sort(self.signal)
        cdf = np.arange(1, len(self.signal) + 1) / len(self.signal)
        plt.plot(ecdf, cdf)
        plt.title("Cumulative Distribution Function (CDF)")
        plt.xlabel("Signal Amplitude")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.show()

# viz
    def plot_spectrogram(self):
        """Plot the spectrogram of the signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        plt.specgram(self.signal, Fs=self.sample_rate)
        plt.title("Spectrogram")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label='Intensity [dB]')
        plt.show()

    def plot_histogram(self, bins=30):
        """Plot a histogram of the signal values."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        plt.hist(self.signal, bins=bins, density=True)
        plt.title("Histogram of Signal Values")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()
      
    def plot_signal(self):
        """Plot the generated signal."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        plt.plot(self.time, self.signal)
        plt.title("Generated Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def plot_psd(self):
        """Plot the Power Spectral Density (PSD) of the signal."""
        frequencies, psd = self.calculate_power_spectral_density()
        plt.semilogy(frequencies, psd)
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power/Frequency [dB/Hz]")
        plt.grid(True)
        plt.show()

    def plot_autocorrelation(self):
        """Plot the auto-correlation of the signal."""
        time_lags, autocorr = self.calculate_autocorrelation()
        plt.plot(time_lags, autocorr)
        plt.title("Auto-correlation")
        plt.xlabel("Time Lag [s]")
        plt.ylabel("Auto-correlation")
        plt.grid(True)
        plt.show()

    def plot_cross_correlation(self, other_signal):
        """Plot the cross-correlation of the signal with another signal."""
        lags, crosscorr = self.calculate_cross_correlation(other_signal)
        plt.plot(lags, crosscorr)
        plt.title("Cross-correlation")
        plt.xlabel("Time Lag [s]")
        plt.ylabel("Cross-correlation")
        plt.grid(True)
        plt.show()

# utils
    def normalize_signal(self):
        """Normalize the signal to have zero mean and unit variance."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        self.signal = (self.signal - np.mean(self.signal)) / np.std(self.signal)
        return self.signal

    def difference_signal(self):
        """Compute the difference between consecutive signal values."""
        if self.signal is None:
            raise ValueError("Signal is not generated yet. Please generate a signal first.")
        self.signal = np.diff(self.signal)
        self.time = self.time[1:]  # Adjust time vector to match signal length
        return self.signal

    def import_from_csv(self, file_path, time_column=0, signal_column=1, delimiter=','):
        """
        Import signal data from a CSV file.

        :param file_path: Path to the CSV file
        :param time_column: Column index for time data
        :param signal_column: Column index for signal data
        :param delimiter: Delimiter used in the CSV file (default is comma)
        :return: Tuple of time and signal arrays
        """
        time_data = []
        signal_data = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row in reader:
                time_data.append(float(row[time_column]))
                signal_data.append(float(row[signal_column]))

        self.time = np.array(time_data)
        self.signal = np.array(signal_data)
        self.sample_rate = 1 / np.mean(np.diff(self.time))  # Calculate sample rate from time data
        self.duration = self.time[-1] - self.time[0]

        return self.time, self.signal

    def import_from_mat(self, file_path, signal_variable_name='signal', time_variable_name='time'):
        """
        Import signal data from a MATLAB .mat file.

        :param file_path: Path to the .mat file
        :param signal_variable_name: Variable name for the signal in the .mat file
        :param time_variable_name: Variable name for the time data in the .mat file
        :return: Tuple of time and signal arrays
        """
        mat_data = scipy.io.loadmat(file_path)
        self.signal = np.array(mat_data[signal_variable_name]).flatten()
        self.time = np.array(mat_data[time_variable_name]).flatten()
        self.sample_rate = 1 / np.mean(np.diff(self.time))  # Calculate sample rate from time data
        self.duration = self.time[-1] - self.time[0]

        return self.time, self.signal

    def import_from_wav(self, file_path):
        """
        Import signal data from a WAV file.

        :param file_path: Path to the WAV file
        :return: Tuple of time and signal arrays
        """
        self.sample_rate, self.signal = scipy.io.wavfile.read(file_path)
        self.signal = self.signal.astype(float)  # Convert to float for processing
        self.duration = len(self.signal) / self.sample_rate
        self.time = np.linspace(0, self.duration, len(self.signal), endpoint=False)

        return self.time, self.signal

    def import_from_hdf5(self, file_path, 
                         signal_dataset_name='signal',
                         time_dataset_name='time'):
        """
        Import signal data from an HDF5 file.

        :param file_path: Path to the HDF5 file
        :param signal_dataset_name: Dataset name for the signal in the HDF5 file
        :param time_dataset_name: Dataset name for the time data in the HDF5 file
        :return: Tuple of time and signal arrays
        """
        with h5py.File(file_path, 'r') as hdf:
            self.signal = np.array(hdf[signal_dataset_name])
            self.time = np.array(hdf[time_dataset_name])
            self.sample_rate = 1 / np.mean(np.diff(self.time))  # Calculate sample rate from time data
            self.duration = self.time[-1] - self.time[0]

        return self.time, self.signal

