from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================
# Rutas
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Original": (
        BASE_DIR
        / "datasets_procesados"
        / "features_original.csv"
    ),
    "Pasa bajos": (
        BASE_DIR
        / "datasets_procesados"
        / "features_pasabajos.csv"
    ),
    "Pasa altos": (
        BASE_DIR
        / "datasets_procesados"
        / "features_pasaaltos.csv"
    )
}

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "comparacion_modelos"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Modelos
# =========================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

svm = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        )
    )
])

modelos = {
    "Random Forest": random_forest,
    "SVM": svm
}


# =========================================
# Validación cruzada
# =========================================

validacion = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

metricas = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro"
}

resultados = []


# =========================================
# Evaluación
# =========================================

for nombre_dataset, ruta_dataset in DATASETS.items():

    print(
        f"\n================================="
        f"\nDataset: {nombre_dataset}"
        f"\n================================="
    )

    df = pd.read_csv(
        ruta_dataset
    )

    X = df.drop(
        columns=["genre", "file"]
    )

    y = df["genre"]

    for nombre_modelo, modelo in modelos.items():

        print(
            f"\nEvaluando {nombre_modelo}..."
        )

        evaluacion = cross_validate(
            modelo,
            X,
            y,
            cv=validacion,
            scoring=metricas,
            n_jobs=-1
        )

        accuracy_promedio = np.mean(
            evaluacion["test_accuracy"]
        )

        accuracy_desviacion = np.std(
            evaluacion["test_accuracy"]
        )

        f1_promedio = np.mean(
            evaluacion["test_f1_macro"]
        )

        f1_desviacion = np.std(
            evaluacion["test_f1_macro"]
        )

        resultados.append({
            "dataset": nombre_dataset,
            "modelo": nombre_modelo,
            "accuracy_promedio": accuracy_promedio,
            "accuracy_desviacion": accuracy_desviacion,
            "f1_macro_promedio": f1_promedio,
            "f1_macro_desviacion": f1_desviacion
        })

        print(
            f"Accuracy: "
            f"{accuracy_promedio:.3f} "
            f"± {accuracy_desviacion:.3f}"
        )

        print(
            f"F1 macro: "
            f"{f1_promedio:.3f} "
            f"± {f1_desviacion:.3f}"
        )


# =========================================
# Guardar resultados
# =========================================

df_resultados = pd.DataFrame(
    resultados
)

ruta_csv = (
    OUTPUT_DIR
    / "resultados_filtrado.csv"
)

df_resultados.to_csv(
    ruta_csv,
    index=False
)

print("\nResultados completos:")
print(df_resultados)

print(
    f"\nCSV guardado en: {ruta_csv}"
)


# =========================================
# Gráfico comparativo
# =========================================

tabla = df_resultados.pivot(
    index="dataset",
    columns="modelo",
    values="accuracy_promedio"
)

errores = df_resultados.pivot(
    index="dataset",
    columns="modelo",
    values="accuracy_desviacion"
)

tabla.plot(
    kind="bar",
    yerr=errores,
    figsize=(10, 6),
    capsize=4
)

plt.title(
    "Comparación de clasificación "
    "según el filtrado"
)

plt.xlabel("Dataset")
plt.ylabel("Accuracy promedio")
plt.ylim(0, 1)
plt.xticks(rotation=0)

plt.tight_layout()

ruta_grafico = (
    OUTPUT_DIR
    / "comparacion_filtrado.png"
)

plt.savefig(ruta_grafico)
plt.show()
plt.close()

print(
    f"Gráfico guardado en: {ruta_grafico}"
)