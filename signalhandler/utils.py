import csv

import h5py
import numpy as np
import scipy





def difference_signal(signal=None, time=None):
    """Compute the difference between consecutive signal values."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    signal = np.diff(signal)
    time = time[1:]  # Adjust time vector to match signal length
    return time, signal


def import_from_csv( file_path, time_column=0, signal_column=1, delimiter=','):
    """  Import signal data from a CSV file.
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

    time = np.array(time_data)
    signal = np.array(signal_data)
    sample_rate = 1 / np.mean(np.diff(time))  # Calculate sample rate from time data
    duration = time[-1] - time[0]

    return time, signal, sample_rate, duration


def import_from_mat( file_path, signal_variable_name='signal', time_variable_name='time'):
    """
    Import signal data from a MATLAB .mat file.

    :param file_path: Path to the .mat file
    :param signal_variable_name: Variable name for the signal in the .mat file
    :param time_variable_name: Variable name for the time data in the .mat file
    :return: Tuple of time and signal arrays
    """
    mat_data = scipy.io.loadmat(file_path)
    signal = np.array(mat_data[signal_variable_name]).flatten()
    time = np.array(mat_data[time_variable_name]).flatten()
    sample_rate = 1 / np.mean(np.diff(time))  # Calculate sample rate from time data
    duration = time[-1] - time[0]

    return time, signal, sample_rate, duration


def import_from_wav( file_path):
    """
    Import signal data from a WAV file.

    :param file_path: Path to the WAV file
    :return: Tuple of time and signal arrays
    """
    sample_rate, signal = scipy.io.wavfile.read(file_path)
    signal = signal.astype(float)  # Convert to float for processing
    duration = len(signal) / sample_rate
    time = np.linspace(0, duration, len(signal), endpoint=False)

    return time, signal


def import_from_hdf5( file_path,
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
        signal = np.array(hdf[signal_dataset_name])
        time = np.array(hdf[time_dataset_name])
        sample_rate = 1 / np.mean(np.diff(time))  # Calculate sample rate from time data
        duration = time[-1] - time[0]

    return time, signal, sample_rate, duration

