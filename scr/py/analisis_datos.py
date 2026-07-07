# -*- coding: utf-8 -*-
"""
.py creado exclusivamente para el analisis de los datos, 
para la realizacion del documento escrito
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../datos/Customer_Sentiment.csv")

print(df.shape)
print(df.dtypes)
print(df.isna().sum())

#Vemos que el data cuenta con 25000 filas y 13 columnas, de las cuales customer_id, customer_rating y response_time_hours son enteros, el resto como texto.

#Veamos como se distribuye la etiqueta de sentimiento
conteo = df["sentiment"].value_counts()
print(conteo)
print((conteo / len(df) * 100).round(1))

porcentaje = (conteo / len(df) * 100).round(1)

plt.bar(porcentaje.index, porcentaje.values, color="#7fb3d9", edgecolor="black")
for i, v in enumerate(porcentaje.values):
    plt.text(i, v + 0.5, f"{v}%", ha="center")
plt.xlabel("Sentimiento")
plt.ylabel("Porcentaje (%)")
plt.title("Distribución de la etiqueta de sentimiento")
plt.ylim(0, 45)
plt.savefig("dist_sentimiento.png", dpi=300, bbox_inches="tight")
plt.show()

#Ahora, veamos la clasificacion promedio por sentimiento

promedio = df.groupby("sentiment")["customer_rating"].mean().round(2)
print(promedio)

plt.bar(promedio.index, promedio.values, color="#7fb3d9", edgecolor="black")
for i, v in enumerate(promedio.values):
    plt.text(i, v + 0.1, f"{v}", ha="center")
plt.xlabel("Sentimiento")
plt.ylabel("Calificación promedio")
plt.title("Calificación promedio por sentimiento")
plt.ylim(0, 5.5)
plt.savefig("calif_por_sentimiento.png", dpi=300, bbox_inches="tight")
plt.show()

#Veamos la dist de sentimientos por categoria de productos

tabla = pd.crosstab(df["product_category"], df["sentiment"], normalize="index") * 100
tabla = tabla.round(1)
print(tabla)

tabla.plot(kind="bar", color=["#d97f7f", "#d9d27f", "#7fb3d9"], edgecolor="black", figsize=(10, 5))
plt.xlabel("Categoría de producto")
plt.ylabel("Porcentaje (%)")
plt.title("Distribución de sentimiento por categoría de producto")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Sentimiento")
plt.tight_layout()
plt.savefig("sentimiento_por_categoria.png", dpi=300, bbox_inches="tight")
plt.show()


#Analisis por genero

genero = pd.crosstab(df["gender"], df["sentiment"], normalize="index") * 100
print(genero.round(1))
print(df.groupby("gender")["customer_rating"].mean().round(2))

#Quejas por servicio

quejas = pd.crosstab(df["sentiment"], df["complaint_registered"], normalize="index") * 100
print(quejas.round(1))
print(df.groupby("sentiment")["response_time_hours"].mean().round(1))

print(df["review_text"].nunique())
print(df["review_text"].value_counts())