import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

canciones = [
    ("blues", "blues.00000.wav"),
    ("classical", "classical.00000.wav"),
    ("jazz", "jazz.00000.wav"),
    ("metal", "metal.00000.wav"),
    ("rock", "rock.00000.wav")
]

for genero, archivo in canciones:

    audio, sr = librosa.load(
        f"Data/genres_original/{genero}/{archivo}",
        sr=None
    )

    # STFT
    stft = librosa.stft(audio)

    # Magnitud en dB
    espectrograma = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    plt.figure(figsize=(10, 5))

    librosa.display.specshow(
        espectrograma,
        sr=sr,
        x_axis="time",
        y_axis="log"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Espectrograma - {genero}")

    plt.tight_layout()

    os.makedirs(f"espectogramas_generados/{genero}", exist_ok=True)
    
    n = len(archivo)
    name = archivo[0:n-4]

    plt.savefig(f"espectogramas_generados/{genero}/espectrograma_{name}.png")

    plt.show()