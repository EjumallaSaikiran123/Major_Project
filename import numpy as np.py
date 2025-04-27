import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display
from scipy.io import wavfile

# Load the WAV file
file_path = 'C:/Users/ejuma_jrrzjzq/Downloads/A voice update/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/YAF_disgust/YAF_love_disgust.wav'  # Replace with your .wav path
sample_rate, data = wavfile.read(file_path)

# If stereo, convert to mono
if len(data.shape) == 2:
    data = np.mean(data, axis=1)

# Normalize
data = data / np.max(np.abs(data))

# Plot 1: Spectrogram
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max),
                         sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

# Plot 2: Wavelet Transform Plot (Continuous Wavelet Transform)
plt.subplot(2, 1, 2)
plt.title("Wavelet Transform (CWT)")
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(data, scales, 'morl', sampling_period=1/sample_rate)
plt.imshow(np.abs(coefficients), extent=[0, len(data) / sample_rate, 1, 128],
           cmap='coolwarm', aspect='auto', vmax=np.max(np.abs(coefficients)) * 0.5)
plt.ylabel('Scale')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()
