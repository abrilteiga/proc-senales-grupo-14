from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# =========================================
# Configuración de rutas
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "genres_original"
OUTPUT_PATH = BASE_DIR / "datasets_procesados" / "features.csv"

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Configuración del procesamiento
# =========================================

# True: procesa las canciones indicadas en CANCIONES_PRUEBA.
# False: procesa todo el dataset.
MODO_PRUEBA = False

CANCIONES_PRUEBA = [
    ("blues", "blues.00000.wav"),
    ("classical", "classical.00000.wav"),
    ("jazz", "jazz.00000.wav"),
    ("metal", "metal.00000.wav"),
]


# =========================================
# Extracción de características
# =========================================

def extraer_caracteristicas(
    ruta_audio: Path,
    genero: str,
    archivo: str
) -> dict:
    """
    Carga un archivo de audio y devuelve sus características
    espectrales y temporales resumidas mediante el promedio.
    """

    audio, sr = librosa.load(
        ruta_audio,
        sr=None
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            y=audio
        )
    )

    rms = np.mean(
        librosa.feature.rms(
            y=audio
        )
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_means = np.mean(
        mfccs,
        axis=1
    )

    fila = {
        "genre": genero,
        "file": archivo,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "zcr": zcr,
        "rms": rms,
    }

    for indice, valor in enumerate(
        mfcc_means,
        start=1
    ):
        fila[f"mfcc_{indice}"] = valor

    return fila


# =========================================
# Obtener archivos a procesar
# =========================================

def obtener_audios() -> list[tuple[str, str, Path]]:
    audios = []

    if MODO_PRUEBA:
        for genero, archivo in CANCIONES_PRUEBA:
            ruta = (
                DATASET_PATH
                / genero
                / archivo
            )

            audios.append(
                (genero, archivo, ruta)
            )

        return audios

    for carpeta_genero in sorted(
        DATASET_PATH.iterdir()
    ):
        if not carpeta_genero.is_dir():
            continue

        genero = carpeta_genero.name

        for ruta_audio in sorted(
            carpeta_genero.glob("*.wav")
        ):
            audios.append(
                (
                    genero,
                    ruta_audio.name,
                    ruta_audio
                )
            )

    return audios


# =========================================
# Programa principal
# =========================================

def main():
    resultados = []
    audios = obtener_audios()

    print(
        f"Cantidad de archivos a procesar: {len(audios)}"
    )

    for genero, archivo, ruta in audios:
        try:
            print(
                f"Procesando {genero}/{archivo}"
            )

            fila = extraer_caracteristicas(
                ruta_audio=ruta,
                genero=genero,
                archivo=archivo
            )

            resultados.append(fila)

        except Exception as error:
            print(
                f"Error procesando {archivo}: {error}"
            )

    df = pd.DataFrame(resultados)

    print("\nPrimeras filas:")
    print(df.head())

    print(
        "\nCantidad de canciones procesadas:",
        len(df)
    )

    if MODO_PRUEBA:
        print(
            "\nModo prueba activado. "
            "No se sobrescribió features.csv."
        )
        print(df)

    else:
        df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print(
            f"\nCSV guardado en: {OUTPUT_PATH}"
        )


if __name__ == "__main__":
    main()