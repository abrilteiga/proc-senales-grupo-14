from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import correlate


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIRS = [
    BASE_DIR / "Data" / "genres_original",
    BASE_DIR / "data" / "genres_original",
]

OUTPUT_DIR = BASE_DIR / "outputs" / "correlacion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGUNDOS_ANALISIS = 5


def obtener_dataset_dir() -> Path:
    for ruta in DATASET_DIRS:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        "No se encontró el dataset en Data/genres_original ni data/genres_original."
    )


def cargar_segmento(ruta_audio: Path, segundos: int) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(ruta_audio, sr=None)
    muestras = min(len(audio), segundos * sr)
    segmento = audio[:muestras]
    segmento = segmento - np.mean(segmento)

    norma = np.linalg.norm(segmento)
    if norma > 0:
        segmento = segmento / norma

    return segmento, sr


def graficar_autocorrelacion(audio: np.ndarray, sr: int, titulo: str, salida: Path) -> None:
    autocorr = correlate(audio, audio, mode="full")
    lags = np.arange(-len(audio) + 1, len(audio)) / sr

    plt.figure(figsize=(10, 4))
    plt.plot(lags, autocorr, color="steelblue")
    plt.title(titulo)
    plt.xlabel("Lag (s)")
    plt.ylabel("Correlación")
    plt.xlim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(salida)
    plt.close()


def analizar_par(
    etiqueta: str,
    ruta_a: Path,
    ruta_b: Path,
    salida_grafico: Path,
) -> dict:
    audio_a, sr_a = cargar_segmento(ruta_a, SEGUNDOS_ANALISIS)
    audio_b, sr_b = cargar_segmento(ruta_b, SEGUNDOS_ANALISIS)

    if sr_a != sr_b:
        raise ValueError(
            f"Las frecuencias de muestreo no coinciden: {sr_a} vs {sr_b}"
        )

    corr = correlate(audio_a, audio_b, mode="full")
    lags = np.arange(-len(audio_b) + 1, len(audio_a)) / sr_a

    indice_max = int(np.argmax(np.abs(corr)))
    lag_max = lags[indice_max]
    corr_max = corr[indice_max]

    plt.figure(figsize=(10, 4))
    plt.plot(lags, corr, color="darkorange")
    plt.axvline(lag_max, color="black", linestyle="--", linewidth=1)
    plt.title(etiqueta)
    plt.xlabel("Lag (s)")
    plt.ylabel("Correlación cruzada")
    plt.xlim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(salida_grafico)
    plt.close()

    return {
        "comparacion": etiqueta,
        "archivo_a": ruta_a.name,
        "archivo_b": ruta_b.name,
        "lag_maximo_segundos": lag_max,
        "correlacion_maxima": corr_max,
    }


def main() -> None:
    dataset_dir = obtener_dataset_dir()

    blues_a = dataset_dir / "blues" / "blues.00000.wav"
    blues_b = dataset_dir / "blues" / "blues.00001.wav"
    metal_a = dataset_dir / "metal" / "metal.00000.wav"

    audio_blues, sr_blues = cargar_segmento(blues_a, SEGUNDOS_ANALISIS)
    graficar_autocorrelacion(
        audio_blues,
        sr_blues,
        "Autocorrelación - blues.00000.wav (primeros 5 segundos)",
        OUTPUT_DIR / "autocorrelacion_blues_00000.png",
    )

    resultados = [
        analizar_par(
            "Correlación cruzada - mismo género (blues vs blues)",
            blues_a,
            blues_b,
            OUTPUT_DIR / "correlacion_mismo_genero.png",
        ),
        analizar_par(
            "Correlación cruzada - distinto género (blues vs metal)",
            blues_a,
            metal_a,
            OUTPUT_DIR / "correlacion_distinto_genero.png",
        ),
    ]

    df = pd.DataFrame(resultados)
    df.to_csv(OUTPUT_DIR / "resumen_correlacion.csv", index=False)

    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
