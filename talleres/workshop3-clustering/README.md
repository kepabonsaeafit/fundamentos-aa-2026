# Workshop 3 - Clustering (K-Means vs. Agglomerative vs. DBSCAN)

Segmentacion de clientes de un centro comercial con tres metodos de clustering, y despues uso de
esos grupos como etiqueta para entrenar un clasificador (no supervisado -> supervisado).

**Dataset:** Mall Customer Segmentation Data - https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## Equipo
- Kevin Pabon Nino - @kepabonsaeafit
- Jose Alejandro Jimenez - @ll333ll
- Geronimo Montes Acebedo - @gmonsteaa

## Notebook

`workshop3_clustering_mall.ipynb` - un solo notebook con todas las fases. Esta ejecutado: las
tablas y graficas se ven en GitHub sin correr nada.

## Datos

El CSV no se versiona. Se descarga en la carpeta `data/` de la raiz del repo:

```bash
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data --unzip
```

El notebook lo lee como `../../data/Mall_Customers.csv`. La version actual del CSV trae la columna
`Gender`; el notebook la renombra a `Genre` para usar los mismos nombres del enunciado.

## Checklist (basado en el enunciado)

- [x] Fase 0 (punto 2) - Exploracion, quitar `CustomerID`, codificar `Genre`, estandarizar
- [x] Fase 1 (punto 3) - K-Means: codo + Silhouette para k = 2..10, k elegido
- [x] Fase 2 (punto 4) - Agglomerative (ward): dendrograma + Silhouette para k = 2..10, k elegido
- [x] Fase 3 (punto 5) - DBSCAN: k-distancias, grilla eps x min_samples, parametros elegidos
- [x] Fase 4 (punto 6) - PCA a 3 componentes y grafico 3D por metodo
- [x] Fase 5 (punto 7) - Tabla comparativa final con Silhouette
- [x] Fase 6 (punto 8) - Clases naturales: cuantos grupos y con que metodo
- [x] Fase 7 (punto 9) - Perfilado de clusters (medias, medianas, boxplots, "personas")
- [x] Fase 8 (punto 10) - Cluster como variable objetivo
- [x] Fase 9 (punto 11) - Train/test estratificado + clasificador
- [x] Fase 10 (punto 12) - Accuracy, classification report, matriz de confusion
- [x] Reflexion final (punto 13)

## Reparto

| Fase | Responsable |
| --- | --- |
| 0 a 4 (puntos 2-6) | Kevin Pabon Nino |
| 5 a 10 + reflexion (puntos 7-13) | Geronimo Montes Acebedo |

## Handoff a la Fase 5

Lo elegido en las fases 0-4:

| Metodo | Parametros | N. clusters | Ruido | Silhouette |
| --- | --- | ---: | ---: | ---: |
| K-Means | k = 6 | 6 | N/A | 0.331 |
| Agglomerative | k = 6, ward | 6 | N/A | 0.310 |
| DBSCAN | eps = 0.9, min_samples = 8 | 4 | 25 (12.5 %) | 0.269 (sin ruido) |

Variables listas en el notebook: `df` (sin `CustomerID`, con `Genre_encoded`), `features`,
`X_scaled`, `scaler`, `tabla_kmeans`, `tabla_agg`, `tabla_dbscan`, `labels_kmeans`, `labels_agg`,
`labels_dbscan`, `X_pca` y `plot_clusters_3d`.

Tres cosas a tener en cuenta al continuar:

1. **El Silhouette no tiene pico** en K-Means ni en Agglomerative: sube siempre hasta k = 10. Por
   eso k = 6 se eligio por el codo, el corte del dendrograma y que los grupos se puedan describir,
   no por el puntaje mas alto.
2. **El genero pesa mucho.** Despues de escalar, varios grupos de K-Means quedan solo de hombres o
   solo de mujeres, y en DBSCAN ningun grupo mezcla generos (las 103 mujeres quedan en un solo
   grupo). Vale la pena comentarlo en la Fase 6.
3. **En DBSCAN el ruido es `-1`.** Si se usa DBSCAN en la Fase 7 o como target, hay que decidir
   que hacer con esos 25 clientes.

## Resultado de las fases 5 a 10

| Decision | Resultado |
| --- | --- |
| Metodo y numero de grupos | K-Means, k = 6 (Silhouette 0.331) |
| Perfiles | 4 grupos por genero y edad (ingreso 40-58 mil) y 2 de ingreso alto (~90 mil): uno gasta mucho (4) y otro poco (5) |
| Cluster que mas llama la atencion | 5: ingreso alto y Spending Score de 18 |
| Clasificador | SVM en Pipeline con StandardScaler, parametros con GridSearchCV (como en la lecture 9): kernel lineal, C = 1, 0.988 en CV sobre train |
| Desempeno | Accuracy 0.95 en test (38 de 40); los 2 errores son clientes en el borde entre dos grupos |

El accuracy alto dice que las fronteras entre clusters son rectas (asi los define K-Means), no
que los grupos esten separados por espacios vacios: el Silhouette sigue en 0.33. El detalle y la
reflexion final estan al final del notebook.

**Nota:** sustentacion aleatoria -> (Sustentacion+Entrega)/2
