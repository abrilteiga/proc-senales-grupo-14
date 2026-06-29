from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt


# =========================================
# Rutas
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "genres_original"
)

OUTPUT_DIR = (
    BASE_DIR
    / "datasets_procesados"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Parámetros del filtro
# =========================================

FRECUENCIA_CORTE = 2000  # Hz
ORDEN_FILTRO = 4


# =========================================
# Filtrado
# =========================================

def aplicar_filtro(
    audio: np.ndarray,
    sr: int,
    tipo: str
) -> np.ndarray:
    """
    Aplica un filtro Butterworth IIR.

    tipo:
        "lowpass"  -> deja pasar frecuencias menores a 2000 Hz.
        "highpass" -> deja pasar frecuencias mayores a 2000 Hz.
    """

    frecuencia_nyquist = sr / 2

    frecuencia_normalizada = (
        FRECUENCIA_CORTE
        / frecuencia_nyquist
    )

    if not 0 < frecuencia_normalizada < 1:
        raise ValueError(
            "La frecuencia de corte debe ser menor "
            "que la frecuencia de Nyquist."
        )

    sos = butter(
        N=ORDEN_FILTRO,
        Wn=frecuencia_normalizada,
        btype=tipo,
        output="sos"
    )

    audio_filtrado = sosfiltfilt(
        sos,
        audio
    )

    return audio_filtrado


# =========================================
# Extracción de características
# =========================================

def extraer_caracteristicas(
    audio: np.ndarray,
    sr: int,
    genero: str,
    archivo: str
) -> dict:

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
        "rms": rms
    }

    for indice, valor in enumerate(
        mfcc_means,
        start=1
    ):
        fila[f"mfcc_{indice}"] = valor

    return fila


# =========================================
# Procesamiento completo
# =========================================

def procesar_dataset(
    tipo_filtro: str,
    nombre_archivo_salida: str
):
    resultados = []

    for carpeta_genero in sorted(
        DATASET_PATH.iterdir()
    ):
        if not carpeta_genero.is_dir():
            continue

        genero = carpeta_genero.name

        print(
            f"\nProcesando género: {genero}"
        )

        for ruta_audio in sorted(
            carpeta_genero.glob("*.wav")
        ):
            try:
                audio, sr = librosa.load(
                    ruta_audio,
                    sr=None
                )

                audio_filtrado = aplicar_filtro(
                    audio=audio,
                    sr=sr,
                    tipo=tipo_filtro
                )

                fila = extraer_caracteristicas(
                    audio=audio_filtrado,
                    sr=sr,
                    genero=genero,
                    archivo=ruta_audio.name
                )

                resultados.append(fila)

            except Exception as error:
                print(
                    f"Error en {ruta_audio.name}: "
                    f"{error}"
                )

    df = pd.DataFrame(resultados)

    ruta_salida = (
        OUTPUT_DIR
        / nombre_archivo_salida
    )

    df.to_csv(
        ruta_salida,
        index=False
    )

    print(
        f"\nArchivo generado: {ruta_salida}"
    )

    print(
        "Canciones procesadas:",
        len(df)
    )


# =========================================
# Programa principal
# =========================================

def main():

    procesar_dataset(
        tipo_filtro="lowpass",
        nombre_archivo_salida=(
            "features_pasabajos.csv"
        )
    )

    procesar_dataset(
        tipo_filtro="highpass",
        nombre_archivo_salida=(
            "features_pasaaltos.csv"
        )
    )


if __name__ == "__main__":
    main()