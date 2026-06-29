import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#Objetivo: Responder qué características fueron las más importantes 
# para que el Random Forest clasificara los géneros musicales.


# ==========================
# Cargar dataset
# ==========================

df = pd.read_csv("features.csv")

# Variables predictoras
X = df.drop(columns=["genre", "file"])

# Etiquetas
y = df["genre"]

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Entrenar Random Forest
# ==========================

modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

modelo.fit(X_train, y_train)

# ==========================
# Importancia de características
# ==========================

importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
)

importancias = importancias.sort_values(ascending=False)

print(importancias)

# ==========================
# Gráfico
# ==========================

plt.figure(figsize=(10,7))

importancias.plot(kind="bar")

plt.xlabel("Características")
plt.ylabel("Importancia")

top10 = importancias.head(10)

plt.figure(figsize=(10,6))
top10.plot(kind="bar")

plt.title("10 características más importantes")

plt.tight_layout()
plt.show()