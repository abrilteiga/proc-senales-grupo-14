import librosa
import matplotlib.pyplot as plt

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

    print("Frecuencia de muestreo:", sr)
    print("Cantidad de muestras:", len(audio))
    print("Duración:", len(audio)/sr)

    plt.figure(figsize=(12,4))
    plt.plot(audio)
    plt.title(f"{genero}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.show()



