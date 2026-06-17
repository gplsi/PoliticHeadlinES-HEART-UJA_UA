# PoliticHeadlinES 2026 — Encoder-based headline ranking

Este repositorio implementa un sistema para **ordenar diez titulares candidatos según su adecuación a un artículo de prensa política en español**. A partir del texto del artículo —y, opcionalmente, de su imagen asociada— el código genera un ranking completo desde el titular más relevante hasta el menos relevante.

El repositorio permite reproducir tres flujos principales:

1. **Ranking por similitud semántica**, usando directamente la similitud entre los embeddings del artículo y de cada titular.
2. **Entrenamiento de un ranker neuronal**, que aprende a ordenar los titulares a partir de rankings correctos incluidos en los datos de entrenamiento.
3. **Predicción con un ranker ya entrenado**, para aplicar un checkpoint existente sobre un nuevo CSV.

También permite comparar distintas formas de representar artículos largos y añadir información visual mediante una fusión tardía de embeddings de texto e imagen.

## 1. Cómo funciona

### 1.1. Codificación del texto

El artículo y los diez titulares se convierten en embeddings con un encoder de Hugging Face. Por defecto se utiliza:

```text
intfloat/multilingual-e5-large
```

Los artículos se procesan con una longitud máxima de **512 tokens** y los titulares con una longitud máxima de **69 tokens**. Los embeddings se normalizan con norma L2 antes de utilizarlos.

El encoder permanece congelado: el código no modifica sus parámetros durante el entrenamiento del ranker.

### 1.2. Ranking por similitud

En el modo `similarity`, cada titular se compara con el artículo mediante el producto escalar entre embeddings normalizados, equivalente a la similitud coseno. Los titulares se ordenan de mayor a menor similitud.

Este modo no entrena ningún modelo y funciona como baseline directo.

### 1.3. Ranker neuronal

En el modo `train`, el código entrena un perceptrón multicapa que asigna una puntuación a cada pareja artículo–titular.

Para representar cada pareja, el ranker concatena:

```text
embedding del artículo
embedding del titular
diferencia absoluta entre ambos
producto elemento a elemento entre ambos
```

El entrenamiento es **pairwise**: para cada artículo se crean pares formados por un titular mejor posicionado y otro peor posicionado en el ranking correcto. El modelo aprende a asignar una puntuación superior al mejor titular mediante `MarginRankingLoss`.

Los ejemplos se dividen aleatoriamente en entrenamiento y validación. Por defecto, el 20 % se utiliza para validación. En cada época se muestran la pérdida de entrenamiento y la pérdida de validación, y se conserva el checkpoint con menor pérdida de validación.

### 1.4. Información visual

Cuando `--image-weight` es mayor que cero, las imágenes se codifican con:

```text
openai/clip-vit-large-patch14
```

La representación final del artículo se obtiene combinando el embedding textual y el embedding visual:

```text
article_embedding = text_weight * text_embedding
                  + image_weight * image_embedding
```

El peso textual se calcula automáticamente como:

```text
text_weight = 1 - image_weight
```

Después de la fusión, el embedding resultante vuelve a normalizarse. Si una imagen no existe, se utiliza un vector visual de ceros para ese ejemplo.

### 1.5. Artículos largos

El repositorio ofrece tres estrategias de representación:

- `first`: utiliza únicamente los primeros 512 tokens.
- `token_chunks`: divide el artículo en fragmentos de tokens, opcionalmente solapados.
- `sentence_chunks`: agrupa oraciones consecutivas sin superar el límite de 512 tokens.

Cuando se generan varios fragmentos, sus embeddings pueden combinarse mediante:

- `mean`: media simple.
- `weighted`: media ponderada que da más importancia a los primeros fragmentos.

La ponderación utiliza un decaimiento exponencial fijo de `0.85`:

```text
peso del fragmento i ∝ 0.85^i
```

## 2. Requisitos

Se recomienda Python 3.11.

Instala las dependencias desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```

La primera ejecución descargará automáticamente desde Hugging Face los modelos de texto y, cuando se use información visual, el modelo CLIP.

## 3. Datos de entrada

El código espera archivos CSV con las siguientes columnas:

```text
id
article_body
image_hash
title_1
title_2
...
title_10
```

Para entrenar también se necesita:

```text
y_true
```

`y_true` debe contener el ranking correcto como una secuencia de identificadores separados por espacios, por ejemplo:

```text
t3 t1 t9 t4 t2 t10 t8 t7 t6 t5
```

Cada identificador corresponde a una columna:

```text
t1  -> title_1
t2  -> title_2
...
t10 -> title_10
```

Las imágenes deben almacenarse en `data/images/`. El nombre de cada archivo debe coincidir con el valor de `image_hash`. Se reconocen automáticamente las extensiones `.jpg`, `.jpeg`, `.png` y `.webp`.

El repositorio incluye una muestra pequeña de los datos para comprobar que el flujo funciona. Para ejecutar experimentos completos, sustituye esos archivos por el conjunto de datos correspondiente manteniendo los mismos nombres o indica otras rutas mediante argumentos.

## 4. Comandos principales

Todos los experimentos se ejecutan desde `run.py`:

```bash
python run.py MODO [OPCIONES]
```

Los modos disponibles son `similarity`, `train` y `predict`.

### 4.1. Ranking por similitud semántica

```bash
python run.py similarity
```

Este comando:

1. Lee por defecto `data/dev_public.csv`.
2. Codifica cada artículo y sus diez titulares.
3. Calcula la similitud entre el artículo y cada titular.
4. Ordena los titulares de mayor a menor similitud.
5. Guarda las predicciones en `outputs/run/predictions.csv`.
6. Si el CSV contiene `y_true`, calcula y muestra la métrica PA-nDCG.

No se entrena ni se guarda ningún modelo.

Para aplicar la similitud a otro archivo:

```bash
python run.py similarity --input-csv data/test_public.csv
```

Para guardar el resultado en otra carpeta:

```bash
python run.py similarity \
  --input-csv data/test_public.csv \
  --output-dir outputs/similarity_test
```

### 4.2. Entrenar el ranker y generar predicciones

```bash
python run.py train
```

Este comando:

1. Lee por defecto `data/train_public.csv`.
2. Comprueba que el CSV contiene la columna `y_true`.
3. Calcula los embeddings de artículos y titulares.
4. Divide los artículos en entrenamiento y validación.
5. Genera todas las comparaciones pairwise entre los diez titulares de cada artículo.
6. Entrena el ranker durante 20 épocas por defecto.
7. Guarda el mejor modelo según la pérdida de validación.
8. Carga `data/dev_public.csv` y genera sus rankings con el modelo entrenado.
9. Guarda las predicciones y, si existe `y_true`, muestra PA-nDCG.

Archivos generados:

```text
outputs/run/best_model.pt
outputs/run/predictions.csv
```

Para cambiar el número de épocas:

```bash
python run.py train --epochs 10
```

Para entrenar con otro CSV y evaluar sobre otro archivo:

```bash
python run.py train \
  --train-csv data/train_public.csv \
  --input-csv data/test_public.csv \
  --output-dir outputs/my_run
```

### 4.3. Predecir con un modelo ya entrenado

```bash
python run.py predict --checkpoint outputs/run/best_model.pt
```

Este comando:

1. Lee por defecto `data/dev_public.csv`.
2. Vuelve a calcular sus embeddings con la configuración indicada.
3. Carga el ranker almacenado en el checkpoint.
4. Asigna una puntuación a cada uno de los diez titulares.
5. Los ordena de mayor a menor puntuación.
6. Guarda el resultado en `outputs/run/predictions.csv`.
7. Si el CSV contiene `y_true`, calcula y muestra PA-nDCG.

Para predecir sobre el conjunto de test:

```bash
python run.py predict \
  --input-csv data/test_public.csv \
  --checkpoint outputs/run/best_model.pt \
  --output-dir outputs/test_predictions
```

Es importante utilizar en predicción la misma representación textual y multimodal empleada al entrenar el checkpoint. Por ejemplo, un modelo entrenado con `--image-weight 0.1` debe aplicarse con ese mismo argumento.

## 5. Configuraciones experimentales

### Primeros 512 tokens del artículo

```bash
python run.py train
```

Es la configuración predeterminada. El artículo se trunca a la primera ventana de 512 tokens y no se utilizan imágenes.

### Fusión de 90 % texto y 10 % imagen

```bash
python run.py train --image-weight 0.1
```

El embedding del artículo se forma con peso textual `0.9` y peso visual `0.1`.

### Fusión de 70 % texto y 30 % imagen

```bash
python run.py train --image-weight 0.3
```

### Fusión equilibrada de texto e imagen

```bash
python run.py train --image-weight 0.5
```

### Fragmentos de tokens con solapamiento y media simple

```bash
python run.py train \
  --strategy token_chunks \
  --overlap 64 \
  --pooling mean \
  --image-weight 0.1
```

El artículo se divide en ventanas de hasta 512 tokens. Cada fragmento comparte 64 tokens con el siguiente y la representación final es la media de todos los embeddings.

### Fragmentos de tokens con media ponderada

```bash
python run.py train \
  --strategy token_chunks \
  --overlap 64 \
  --pooling weighted \
  --image-weight 0.1
```

La división es la misma, pero los fragmentos iniciales reciben más peso que los posteriores.

### Fragmentos basados en oraciones

```bash
python run.py train \
  --strategy sentence_chunks \
  --pooling mean \
  --image-weight 0.1
```

Las oraciones se agrupan consecutivamente hasta alcanzar el límite del encoder, sin partir el texto mediante una ventana fija de tokens.

### Fragmentos basados en oraciones con media ponderada

```bash
python run.py train \
  --strategy sentence_chunks \
  --pooling weighted \
  --image-weight 0.1
```

## 6. Argumentos disponibles

| Argumento | Valor por defecto | Descripción |
|---|---:|---|
| `mode` | obligatorio | Operación: `similarity`, `train` o `predict`. |
| `--input-csv` | `data/dev_public.csv` | CSV sobre el que se generan las predicciones. |
| `--train-csv` | `data/train_public.csv` | CSV con `y_true` utilizado para entrenar. |
| `--images-dir` | `data/images` | Carpeta que contiene las imágenes asociadas a los artículos. |
| `--output-dir` | `outputs/run` | Carpeta donde se guardan el checkpoint y las predicciones. |
| `--checkpoint` | `outputs/run/best_model.pt` | Modelo que se carga en el modo `predict`. |
| `--strategy` | `first` | Representación del artículo: `first`, `token_chunks` o `sentence_chunks`. |
| `--pooling` | `mean` | Agregación de fragmentos: `mean` o `weighted`. |
| `--overlap` | `0` | Número de tokens compartidos por fragmentos consecutivos en `token_chunks`. |
| `--image-weight` | `0.0` | Peso de la imagen entre 0 y 1. El peso textual es `1 - image_weight`. |
| `--text-model` | `intfloat/multilingual-e5-large` | Encoder de Hugging Face utilizado para artículos y titulares. |
| `--encoder-batch-size` | `8` | Tamaño de lote durante el cálculo de embeddings textuales. |
| `--ranker-batch-size` | `128` | Tamaño de lote durante el entrenamiento pairwise. |
| `--epochs` | `20` | Número de épocas de entrenamiento. |
| `--seed` | `42` | Semilla para la división de datos y la inicialización del entrenamiento. |

Para consultar esta ayuda desde la terminal:

```bash
python run.py --help
```

## 7. Formato de salida

Las predicciones se guardan en un CSV con tres columnas:

```text
id,task_1,task_2
```

Cada fila contiene un ranking completo, por ejemplo:

```text
123,t3 t1 t9 t4 t2 t10 t8 t7 t6 t5,t3 t1 t9 t4 t2 t10 t8 t7 t6 t5
```

En esta implementación, el mismo ranking se escribe en `task_1` y `task_2`. La diferencia entre una configuración textual y una multimodal depende de si se ejecuta el modelo con `--image-weight 0` o con un peso visual mayor que cero.

## 8. Evaluación

Cuando el archivo de entrada incluye `y_true`, el código calcula automáticamente **PA-nDCG@10** con `alpha = 0.9`.

La métrica exige acertar el titular situado en primera posición. Si el primer titular predicho no coincide con el primero del ranking correcto, la puntuación del ejemplo es cero. Cuando la primera posición es correcta, el resto del ranking contribuye mediante una variante de nDCG.

La puntuación media se muestra en la terminal:

```text
PA-nDCG: 0.XXXXXX
```

## 9. CPU y GPU

No es obligatorio disponer de GPU. El código selecciona automáticamente:

```text
CUDA, si está disponible
CPU, en caso contrario
```

La parte más costosa es el cálculo de embeddings con E5 y, en configuraciones multimodales, con CLIP. La muestra incluida puede ejecutarse en CPU, pero para procesar el conjunto completo se recomienda una GPU con memoria suficiente.

El argumento `--encoder-batch-size` permite reducir el consumo de memoria durante la codificación. Por ejemplo:

```bash
python run.py train --encoder-batch-size 2
```

Si el entrenamiento del ranker consume demasiada memoria, también puede reducirse:

```bash
python run.py train --ranker-batch-size 32
```

## 10. Estructura del repositorio

```text
iberlef_simple/
├── run.py
├── requirements.txt
├── README.md
├── data/
│   ├── README.md
│   ├── train_public.csv
│   ├── dev_public.csv
│   ├── test_public.csv
│   └── images/
├── outputs/
└── src/
    └── politicheadlines/
        ├── __init__.py
        ├── data.py
        ├── encoders.py
        ├── features.py
        ├── metrics.py
        ├── pipeline.py
        └── ranker.py
```

Responsabilidad de cada archivo:

- `run.py`: interfaz de línea de comandos y configuración general.
- `data.py`: lectura y validación de CSV y tratamiento de rankings.
- `encoders.py`: codificación de texto e imágenes y creación de fragmentos.
- `features.py`: construcción de embeddings y fusión texto–imagen.
- `ranker.py`: arquitectura, entrenamiento pairwise, guardado, carga y predicción.
- `metrics.py`: implementación de PA-nDCG.
- `pipeline.py`: coordinación de los modos `similarity`, `train` y `predict`.
- `outputs/`: checkpoints y archivos CSV generados.
