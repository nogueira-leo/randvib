import core.signalhandler as sh

# Example usage:
handler = sh.SignalHandler(sample_rate=1000, duration=5)

# Generate and plot a random vibration
handler.generate_random_vibration(frequency=10, amplitude=5)
handler.plot_signal()

# Plot the Power Spectral Density
handler.plot_psd()

# Apply a high-pass filter and plot the signal
handler.apply_high_pass_filter(cutoff_frequency=15)
handler.plot_signal()

# Plot the auto-correlation
handler.plot_autocorrelation()

# Generate a second signal and plot cross-correlation
handler2 = sh.SignalHandler(sample_rate=1000, duration=5)
handler2.generate_random_vibration(frequency=15, amplitude=3)
handler.plot_cross_correlation(handler2.signal)

# Process the signal in chunks (simulate real-time processing)
for chunk in handler.process_signal_in_chunks(chunk_size=100):
    print("Processing chunk:", chunk)

# Initialize the SignalHandler
handler = sh.SignalHandler()

# Import signal from a CSV file
handler.import_from_csv('signal_data.csv')

# Import signal from a MAT file
handler.import_from_mat('signal_data.mat')

# Import signal from a WAV file
handler.import_from_wav('signal_data.wav')

# Import signal from an HDF5 file
handler.import_from_hdf5('signal_data.h5')

# Now, you can use the imported signal for further processing
handler.plot_signal()
handler.plot_psd()
