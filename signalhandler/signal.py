import numpy as np
import matplotlib.pyplot as plt

import csv
import scipy.io
import h5py
from matplotlib.pyplot import tight_layout
from scipy import fft
from scipy.signal import welch
from signalhandler import viz, stats



class Signal:
    def __init__(self, signal_in=None, sample_rate=1000, duration=10, unit='-'):
        """
        Initialize the SignalHandler with a specific sample rate and duration.

        :param sample_rate: Number of samples per second (Hz)
        :param duration: Duration of the signal in seconds
        """
        self.autocorr = None
        self.Pxx = None
        self.f = None
        self._psd = None
        self.unit = unit
        self.len = int(sample_rate * duration)
        if signal_in is None:
            signal_in = np.zeros(self.len)
        self.signal = np.array(signal_in)
        self.time = np.linspace(0, duration, self.len, endpoint=False)
        self.fs = sample_rate

    def plot(self):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, layout='tight')
        ax0.plot(self.time, self.signal)
        ax0.grid(True)
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel(f'Signal [{self.unit}]')

        ax1.grid(True)
        ax1.psd(self.signal,NFFT= self.len//2,Fs=self.fs)

        ax2.acorr(self.signal, maxlags=None)

        ax2.grid(True)

        plt.show()

    def psd(self):
            self.f, self.Pxx = welch(self.signal, self.fs,
                                     nperseg=self.len, axis=0)




    def add_white_noise(self, mean=0, std=1):
        """Generate a white noise signal."""
        noise = np.random.normal(mean, std, len(self.time))
        self.signal += noise
        return self.signal

    def add_sine(self, frequency=None, amplitude= None):
        """Generate a random vibration signal."""

        sine_wave =amplitude * np.sin(2 * np.pi * frequency * self.time)

        self.signal += sine_wave
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
        dt = 1/self.fs
        x = np.zeros(len(self.time))
        for i in range(1, len(self.time)):
            x[i] = x[i-1] + theta*(mu - x[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal()
        self.signal = x
        return self.signal





# stats


    def calculate_cross_correlation(self, other_signal):
        """Calculate the cross-correlation of the signal with another signal."""
        if self.signal is None or other_signal is None:
            raise ValueError("Both signals must be generated first.")
        crosscorr = np.correlate(self.signal, other_signal, mode='full')
        lags = np.arange(-len(self.signal) + 1, len(self.signal))
        return lags / self.fs, crosscorr
    
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
        self.fs = 1 / np.mean(np.diff(self.time))  # Calculate sample rate from time data
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
        self.fs = 1 / np.mean(np.diff(self.time))  # Calculate sample rate from time data
        self.duration = self.time[-1] - self.time[0]

        return self.time, self.signal

    def import_from_wav(self, file_path):
        """
        Import signal data from a WAV file.

        :param file_path: Path to the WAV file
        :return: Tuple of time and signal arrays
        """
        self.fs, self.signal = scipy.io.wavfile.read(file_path)
        self.signal = self.signal.astype(float)  # Convert to float for processing
        self.duration = len(self.signal) / self.fs
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
            self.fs = 1 / np.mean(np.diff(self.time))  # Calculate sample rate from time data
            self.duration = self.time[-1] - self.time[0]

        return self.time, self.signal
