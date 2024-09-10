from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('qtagg')

def plot_signal(time, signal, title='Signal', units='-'):
    """Plot the generated signal."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(f"Amplitude [{units}]")
    plt.grid(True)
    plt.show()

def plot_freq(signal, title:str, label:str, units:str='-'):
    """Plot the Power Spectral Density (PSD) of the signal."""
    plt.semilogy(signal)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(f'{label} [{units}]')
    plt.grid(True)
    plt.show()

def plot_spectrogram(spect, time, freq):
    """Plot the spectrogram of the signal."""
    if spect is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    plt.imshow(spect, aspect='auto', origin='lower', interpolation='gaussian', extent=(time[0], time[-1], freq[0], freq[-1]))
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label='Intensity [dB]')
    plt.show()

def plot_histogram(signal, bins=30):
    """Plot a histogram of the signal values."""
    if signal is None:
        raise ValueError("Signal is not generated yet. Please generate a signal first.")
    plt.hist(signal, bins=bins, density=True)
    plt.title("Histogram of Signal Values")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()