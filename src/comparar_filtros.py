from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================================
# Rutas
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "datasets_procesados"
)

PASA_ALTOS_FILE = (
    DATASET_DIR
    / "features_pasaaltos.csv"
)

PASA_BAJOS_FILE = (
    DATASET_DIR
    / "features_pasabajos.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "comparacion_filtros"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Cargar archivos
# =========================================

df_altos = pd.read_csv(PASA_ALTOS_FILE)
df_bajos = pd.read_csv(PASA_BAJOS_FILE)

print("Pasa altos:", df_altos.shape)
print("Pasa bajos:", df_bajos.shape)


# =========================================
# Verificar canciones
# =========================================

archivos_altos = set(df_altos["file"])
archivos_bajos = set(df_bajos["file"])

solo_altos = archivos_altos - archivos_bajos
solo_bajos = archivos_bajos - archivos_altos

if solo_altos or solo_bajos:
    print("\nAdvertencia: los archivos no coinciden.")

    print("Solo en pasa altos:")
    print(solo_altos)

    print("Solo en pasa bajos:")
    print(solo_bajos)

else:
    print(
        "\nAmbos datasets contienen "
        "las mismas canciones."
    )


# =========================================
# Características a comparar
# =========================================

caracteristicas_basicas = [
    "centroid",
    "bandwidth",
    "zcr",
    "rms"
]


# =========================================
# Promedios globales
# =========================================

promedio_altos = (
    df_altos[caracteristicas_basicas]
    .mean()
)

promedio_bajos = (
    df_bajos[caracteristicas_basicas]
    .mean()
)

comparacion_global = pd.DataFrame({
    "Pasa altos": promedio_altos,
    "Pasa bajos": promedio_bajos
})

comparacion_global["Diferencia"] = (
    comparacion_global["Pasa altos"]
    - comparacion_global["Pasa bajos"]
)

print("\nPromedios globales:")
print(comparacion_global)

comparacion_global.to_csv(
    OUTPUT_DIR / "comparacion_global.csv"
)


# =========================================
# Promedios por género
# =========================================

promedios_altos_genero = (
    df_altos
    .groupby("genre")[caracteristicas_basicas]
    .mean()
)

promedios_bajos_genero = (
    df_bajos
    .groupby("genre")[caracteristicas_basicas]
    .mean()
)

promedios_altos_genero.to_csv(
    OUTPUT_DIR
    / "promedios_pasaaltos_por_genero.csv"
)

promedios_bajos_genero.to_csv(
    OUTPUT_DIR
    / "promedios_pasabajos_por_genero.csv"
)


# =========================================
# Gráficos por característica
# =========================================

for caracteristica in caracteristicas_basicas:

    comparacion_genero = pd.DataFrame({
        "Pasa altos": (
            promedios_altos_genero[
                caracteristica
            ]
        ),
        "Pasa bajos": (
            promedios_bajos_genero[
                caracteristica
            ]
        )
    })

    comparacion_genero.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title(
        f"Comparación de {caracteristica} "
        "por género"
    )

    plt.xlabel("Género")
    plt.ylabel(caracteristica)
    plt.xticks(rotation=45)

    plt.tight_layout()

    ruta_grafico = (
        OUTPUT_DIR
        / f"comparacion_{caracteristica}.png"
    )

    plt.savefig(ruta_grafico)
    plt.show()
    plt.close()


# =========================================
# Comparación de MFCC
# =========================================

columnas_mfcc = [
    columna
    for columna in df_altos.columns
    if columna.startswith("mfcc_")
]

mfcc_altos = (
    df_altos[columnas_mfcc]
    .mean()
)

mfcc_bajos = (
    df_bajos[columnas_mfcc]
    .mean()
)

comparacion_mfcc = pd.DataFrame({
    "Pasa altos": mfcc_altos,
    "Pasa bajos": mfcc_bajos
})

comparacion_mfcc.to_csv(
    OUTPUT_DIR / "comparacion_mfcc.csv"
)

comparacion_mfcc.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Comparación promedio de MFCC"
)

plt.xlabel("Coeficiente MFCC")
plt.ylabel("Valor promedio")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "comparacion_mfcc.png"
)

plt.show()
plt.close()


print(
    f"\nResultados guardados en: "
    f"{OUTPUT_DIR}"
)