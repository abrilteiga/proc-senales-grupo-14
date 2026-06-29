import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

canciones = [
    ("blues", "blues.00000.wav"),
    ("classical", "classical.00000.wav"),
    ("jazz", "jazz.00000.wav"),
    ("metal", "metal.00000.wav"),
]

for genero, archivo in canciones:

    audio, sr = librosa.load(
        f"Data/genres_original/{genero}/{archivo}",
        sr=None
    )

    N = len(audio)

    X = fft(audio)

    freqs = np.fft.fftfreq(N, d=1/sr)

    plt.figure(figsize=(12, 4))

    plt.plot(
        freqs[:N//2],
        np.abs(X[:N//2])
    )

    plt.title(f"FFT - {genero}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")

    plt.grid(True)

    plt.show()