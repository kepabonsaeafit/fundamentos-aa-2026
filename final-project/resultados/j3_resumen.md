# J3: modelado y seleccion con la base v1_A

Train 2019-2023, test 2024 y deriva 2025, solo contratos maduros. IC 95 % con bootstrap por entidad (200 repeticiones). Regla por entidad con m = 10 (elegido con folds temporales). [VERIFICADO: salida de src/modelado.py y src/evaluar_j3.py]

## Busqueda (folds temporales 2021-2023, PR-AUC medio)

| Modelo | n train | n_iter | Segundos | PR-AUC cv | ROC-AUC cv | Mejores parametros |
|---|---|---|---|---|---|---|
| hgb | 261.249 | 25 | 287 | 0.5765 ± 0.0046 | 0.7620 | `{'l2_regularization': 0.002259279742015696, 'learning_rate': 0.034003508264422166, 'max_features': 0.5, 'max_iter': 200, 'max_leaf_nodes': 127, 'min_samples_leaf': 20}` |
| arbol | 261.249 | 30 | 51 | 0.5318 ± 0.0104 | 0.7321 | `{'min_samples_leaf': 200, 'max_depth': 12, 'criterion': 'entropy', 'class_weight': None}` |
| rf | 261.249 | 12 | 300 | 0.5781 ± 0.0130 | 0.7567 | `{'n_estimators': 400, 'min_samples_leaf': 20, 'max_features': 'sqrt', 'max_depth': None}` |
| knn | 20.000 | 15 | 17 | 0.5169 ± 0.0471 | 0.6925 | `{'weights': 'distance', 'p': 1, 'n_neighbors': 100}` |
| svm | 20.000 | 10 | 45 | 0.5225 ± 0.0407 | 0.6904 | `{'C': 1.5751320499779735, 'gamma': 0.01}` |

## Test 2024 (una sola corrida)

| Modelo | PR-AUC | dif. PR-AUC vs regla | ROC-AUC | ROC-AUC intra-entidad | P@1 % | P@5 % | P@10 % |
|---|---|---|---|---|---|---|---|
| regla_base | 0,247 [0,215; 0,285] | -0,223 [-0,292; -0,129] | 0,500 [0,500; 0,500] | 0,500 [0,500; 0,500] | 0,247 | 0,247 | 0,247 |
| regla_entidad | 0,470 [0,355; 0,562] | +0,000 [0,000; 0,000] | 0,682 [0,628; 0,726] | 0,500 [0,500; 0,500] | 0,840 | 0,738 | 0,594 |
| hgb | 0,584 [0,488; 0,659] | +0,114 [0,082; 0,158] | 0,781 [0,752; 0,807] | 0,701 [0,597; 0,757] | 0,928 | 0,831 | 0,729 |
| arbol | 0,537 [0,441; 0,616] | +0,067 [0,036; 0,113] | 0,751 [0,721; 0,782] | 0,666 [0,560; 0,726] | 0,909 | 0,782 | 0,691 |
| rf | 0,580 [0,481; 0,657] | +0,110 [0,081; 0,148] | 0,779 [0,748; 0,806] | 0,697 [0,597; 0,752] | 0,910 | 0,852 | 0,703 |
| knn | 0,509 [0,392; 0,604] | +0,039 [0,015; 0,065] | 0,719 [0,678; 0,754] | 0,626 [0,584; 0,685] | 0,898 | 0,785 | 0,631 |
| svm | 0,523 [0,404; 0,613] | +0,053 [0,030; 0,085] | 0,730 [0,689; 0,766] | 0,644 [0,595; 0,713] | 0,880 | 0,822 | 0,660 |

Criterio de aceptacion del HGB final (IC de la dif. de PR-AUC excluye 0): **True**

## Categorias nativas frente a codigos ordinales (mismos hiperparametros)

| Variante | PR-AUC cv | ROC-AUC cv | PR-AUC por fold | PR-AUC 2024 |
|---|---|---|---|---|
| ordinal | 0.5765 | 0.7620 | [0.5809, 0.5783, 0.5702] | 0,584 [0,488; 0,659] |
| nativas | 0.5798 | 0.7646 | [0.586, 0.5772, 0.5762] | 0,588 [0,490; 0,663] |

## Ablacion (test 2024)

| Variante | PR-AUC | dif. vs regla | ROC-AUC | ROC-AUC intra (pares) | dif. intra vs regla |
|---|---|---|---|---|---|
| regla_entidad | 0,470 [0,355; 0,562] | +0,000 [0,000; 0,000] | 0,682 [0,628; 0,726] | 0,500 [0,500; 0,500] | +0,000 [0,000; 0,000] |
| final | 0,584 [0,488; 0,659] | +0,114 [0,082; 0,158] | 0,781 [0,752; 0,807] | 0,701 [0,597; 0,757] | +0,201 [0,097; 0,257] |
| sin_ablacion_j3 | 0,580 [0,488; 0,655] | +0,110 [0,079; 0,153] | 0,777 [0,749; 0,804] | 0,698 [0,597; 0,754] | +0,198 [0,097; 0,254] |
| sin_por_verificar | 0,584 [0,489; 0,661] | +0,114 [0,083; 0,157] | 0,781 [0,752; 0,808] | 0,701 [0,597; 0,757] | +0,201 [0,097; 0,257] |

## Deriva 2025 (n = 99.702. tasa base 0.2714)

| Puntaje | PR-AUC | dif. vs regla 2019-2023 | ROC-AUC |
|---|---|---|---|
| regla_base | 0,271 [0,243; 0,305] | -0,194 [-0,253; -0,131] | 0,500 [0,500; 0,500] |
| regla_entidad | 0,465 [0,385; 0,540] | +0,000 [0,000; 0,000] | 0,656 [0,624; 0,691] |
| final_2019_2023 | 0,550 [0,472; 0,642] | +0,084 [0,033; 0,157] | 0,773 [0,747; 0,797] |
| regla_entidad_2019_2024 | 0,467 [0,380; 0,543] | +0,002 [-0,006; 0,008] | 0,664 [0,629; 0,698] |
| final_2019_2024 | 0,550 [0,469; 0,664] | +0,085 [0,029; 0,173] | 0,785 [0,759; 0,811] |

## Sensibilidad: evaluado con y_dias en 2024 (tasa base 0.1209)

| Puntaje | PR-AUC | dif. vs regla (y_dias) | ROC-AUC |
|---|---|---|---|
| regla_base | 0,121 [0,096; 0,152] | -0,169 [-0,223; -0,103] | 0,500 [0,500; 0,500] |
| regla_entidad | 0,290 [0,204; 0,371] | +0,000 [0,000; 0,000] | 0,720 [0,672; 0,758] |
| final_entrenado_con_y | 0,353 [0,260; 0,433] | +0,063 [0,017; 0,113] | 0,765 [0,724; 0,798] |
| final_entrenado_con_y_dias | 0,325 [0,258; 0,383] | +0,035 [-0,008; 0,089] | 0,771 [0,738; 0,795] |

## Entidades no vistas en train (1.834 contratos. 211 positivos)

HGB final: PR-AUC 0,279 [0,170; 0,449], dif. vs regla +0,164 [0,069; 0,307]; ROC-AUC 0,721 [0,655; 0,791], dif. +0,221 [0,155; 0,291].

## SHAP global (TreeExplainer, 3000 contratos de 2024; error de aditividad maximo 7.4e-13)

| Feature | media |SHAP| (log-odds) |
|---|---|
| nit_entidad | 0.6964 |
| p_dur_dias | 0.2607 |
| mes_firma | 0.2227 |
| fin_planeado_cruza_anio | 0.2081 |
| log_precio_base | 0.1911 |
| dias_firma_a_inicio | 0.0875 |
| habilita_pago_adelantado | 0.0697 |
| justificacion_modalidad_de | 0.0517 |
| destino_gasto | 0.0341 |
| departamento | 0.0327 |

Por hipotesis:

| Hipotesis | suma media |SHAP| |
|---|---|
| H6 | 0.6964 |
| H4 | 0.4620 |
| H5 | 0.2822 |
| - | 0.2651 |
| - (tamano del contrato) | 0.1911 |
| H6 (proxy) | 0.1175 |
| H3 | 0.0801 |
| H7 | 0.0543 |
| H2 | 0.0227 |
| H7 (proxy de complejidad) | 0.0159 |
| - (complejidad) | 0.0151 |
| H1 | 0.0114 |
