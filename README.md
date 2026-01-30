# SatSem: Segmentación de Manglares con Sentinel-2

Sistema de segmentación semántica basado en deep learning para la detección y mapeo de manglares utilizando imágenes de teledetección Sentinel-2. El proyecto implementa múltiples arquitecturas de redes neuronales para la clasificación de píxeles en áreas de manglar.

## 📋 Descripción General

SatSem es un pipeline completo que incluye:
- **Entrenamiento** de modelos de segmentación semántica
- **Inferencia** sobre nuevas imágenes satelitales
- **Generación de mosaicos** georreferenciados
- **Cálculo de métricas** de precisión (IoU, Confusión)
- **Post-procesamiento morfológico** para mejorar resultados

**Área de Estudio:** Jambeli (Ecuador)
**Datos:** Imágenes Sentinel-2 (2020-2025) con máscaras manuales

---

## 📁 Estructura de Carpetas

```
SatSem/
├── data/                          # Datos de entrenamiento, validación y prueba
│   ├── archive_shape/             # Datos geométricos del área de estudio
│   │   └── Jambeli_corregido/
│   │       ├── Area_Estudio_Jambeli.shp    # Shapefile del área
│   │       ├── Area_Estudio_Jambeli.shx    # Índice del shapefile
│   │       ├── Area_Estudio_Jambeli.dbf    # Atributos del shapefile
│   │       ├── Area_Estudio_Jambeli.prj    # Proyección (UTM 17S)
│   │       └── Area_Estudio_Jambeli.qmd    # Metadatos adicionales
│   ├── train/                     # Conjunto de entrenamiento (323 pares)
│   │   ├── renaming_mapping.txt   # Mapeo de renumeración + distribución
│   │   └── stratified_mapping.json# JSON con mapeo estratificado
│   │   └── tile_*.tif, mask_*.tif # Imágenes y máscaras
│   ├── val/                       # Conjunto de validación
│   │   ├── renaming_mapping.txt   
│   │   └── stratified_mapping.json
│   │   └── tile_*.tif, mask_*.tif 
│   └── test/                      # Datos de prueba (sin anotaciones)
│       ├── Manglar_2020_images/   # Imágenes satelitales por año
│       ├── Manglar_2020_masks/    # Máscaras de predicción
│       ├── Manglar_2021_images/
│       ├── Manglar_2021_masks/
│       └── ...hasta 2025
├── model/                         # Modelos y código de entrenamiento
│   └── train_multibranch_v_copy_v1.py  # Script de entrenamiento
├── Script/                        # Scripts de inferencia y análisis
│   ├── train_datos_refactored.ipynb     # Notebook interactivo
│   └── predict_and_mosaic_with_metrics.py  # Pipeline de predicción
└── README.md                      # Este archivo
```

---

## 📊 Descripción de Archivos y Directorio

### 📦 Directorio `data/`

#### `data/archive_shape/Jambeli_corregido/`
**Propósito:** Geometría del área de estudio para enmascaramiento y validación espacial

| Archivo | Descripción |
|---------|-----------|
| `Area_Estudio_Jambeli.shp` | Shapefile vectorial (polígono) del área de estudio |
| `Area_Estudio_Jambeli.shx` | Índice de acceso rápido al shapefile |
| `Area_Estudio_Jambeli.dbf` | Base de datos con atributos del polígono |
| `Area_Estudio_Jambeli.prj` | Definición de proyección (UTM zona 17S) |
| `Area_Estudio_Jambeli.qmd` | Metadatos complementarios |

**Uso:** 
```python
# Cargar en predict_and_mosaic_with_metrics.py
gdf = gpd.read_file('data/archive_shape/Jambeli_corregido/Area_Estudio_Jambeli.shp')
# Enmascarar predicciones fuera del área de estudio
```

---

#### `data/train/`
**Propósito:** Conjunto de entrenamiento con 323 pares imagen-máscara estratificados

**Composición:**
- **negative:** 15 teselas (~4.6%) - Sin manglar (0% de cobertura)
- **positive_dense:** 276 teselas (~85.5%) - Alto porcentaje de manglar (>20%)
- **positive_sparse:** 32 teselas (~10%) - Bajo porcentaje de manglar (0-20%)

**Archivos Clave:**

| Archivo | Descripción |
|---------|-----------|
| `stratified_mapping.json` | Mapeo JSON de cada tesela con metadatos |
| `renaming_mapping.txt` | Registro legible de numeración + distribución |
| `tile_0000.tif` a `tile_0322.tif` | 323 imágenes Sentinel-2 (11 bandas, 256×256 px) |
| `mask_0000.tif` a `mask_0322.tif` | 323 máscaras binarias anotadas manualmente |

**Estructura del JSON:**
```json
{
  "new_idx": 0,
  "new_tile": "tile_0000.tif",
  "new_mask": "mask_0000.tif",
  "temp_tile": "temp_tile_0182.tif",    // Nombre temporal original
  "temp_mask": "temp_mask_0182.tif",    // Nombre temporal original
  "tile_type": "positive_dense",        // Categoría de manglar
  "manglar_percentage": 9.735           // Porcentaje de cobertura
}
```

**Propósito del mapeo:** Permite rastrear la correspondencia entre la numeración final y los archivos temporales originales para reproducibilidad.

---

#### `data/val/`
**Propósito:** Conjunto de validación con estructura idéntica a `train/`

**Composición:** Similar a train, pero representa ~20% del dataset total

**Archivos:**
- `stratified_mapping.json` - Mapeo de validación
- `renaming_mapping.txt` - Distribución de validación
- `tile_*.tif` y `mask_*.tif` - Pares de validación

**Uso:** Evaluación de modelo durante entrenamiento (no se entrena con estos datos)

---

#### `data/test/Manglar_20XX_images/` y `data/test/Manglar_20XX_masks/`
**Propósito:** Datos de prueba sin anotaciones para evaluación temporal (2020-2025)

**Estructura por año:**
```
Manglar_2020_images/
├── image_2020_0000.tif      # Imagen Sentinel-2 sin anotar
├── image_2020_0001.tif
└── ...

Manglar_2020_masks/
├── mask_2020_0000.tif       # Predicciones generadas por el modelo
├── mask_2020_0001.tif
└── ...
```

**Propósito:** 
- Evaluación en diferentes años
- Análisis temporal de cambios en manglar
- Validación de rendimiento en datos no vistos
- Generación de mosaicos anuales

---

### 🎯 Directorio `model/`

#### `model/train_multibranch_v_copy_v1.py`
**Propósito:** Script principal de entrenamiento con múltiples arquitecturas

**Características:**

| Componente | Descripción |
|-----------|-----------|
| **Modelos CNN** | UNet, UNet++, DeepLabV3+, PSPNet, HRNet |
| **Multi-Branch UNet++** | Procesamiento dual: resolución alta (10m) + baja (20m) con fusión FPN |
| **SegFormer (Vision Transformer)** | Arquitectura transformer para captura de contexto global |
| **Random Forest** | Baseline de machine learning clásico |
| **Data Module** | Compatible con TorchGeo para gestión de datos |

**Flujo de Ejecución:**
1. **Carga de datos:** Lee tiles y máscaras de `data/train/` y `data/val/`
2. **Análisis de distribución:** Genera histogramas de clases
3. **Configuración del modelo:** Selecciona arquitectura y encoder
4. **Entrenamiento:** Iteraciones con validación periódica
5. **Guardado:** Checkpoint del mejor modelo
6. **Evaluación:** Métricas IoU, Dice, Confusión

**Configuración (modificable en el script):**
```python
MODEL_TYPE = "multi_branch_unet"  # O: "unet", "segformer", "random_forest"
ENCODER = "resnet50"              # Para modelos SMP
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
```

**Salida:**
- Checkpoints de modelo (`.pth`)
- Curvas de pérdida y métrica
- Gráficos de distribución de clases
- Logs de entrenamiento

---

### 📝 Directorio `Script/`

#### `Script/train_datos_refactored.ipynb`
**Propósito:** Notebook Jupyter interactivo para exploración y análisis de datos

**Celdas incluidas:**
1. Importación de librerías
2. Carga y visualización de tiles/máscaras
3. Estadísticas de distribución
4. Análisis espacial y espectral
5. Visualización de muestras
6. Verificación de anotaciones

**Uso:**
```bash
# Ejecutar en Jupyter
jupyter notebook Script/train_datos_refactored.ipynb
```

---

#### `Script/predict_and_mosaic_with_metrics.py`
**Propósito:** Pipeline completo de predicción, mosaicado y evaluación

**Funcionalidades Principales:**

| Función | Descripción |
|---------|-----------|
| **predict_tile()** | Infiere máscara para una tesela individual |
| **merge_tiles()** | Fusiona predicciones de múltiples teselas en mosaico |
| **calculate_metrics()** | Calcula IoU, precisión, recall, F1-score |
| **morphological_postprocessing()** | Aplica erosión/dilatación para limpiar artefactos |
| **reproject_to_utm()** | Reproyecta resultados a coordenadas UTM |
| **clip_to_aoi()** | Recorta mosaicos al área de estudio (shapefile) |

**Optimizaciones Clave (Diciembre 2024):**
- **Sin normalización percentil redundante:** Sentinel-2 ya viene normalizado [0,1] de Google Earth Engine
- **Threshold optimizado:** 0.50 para valores espectrales reales
- **Mejora estimada:** +8-10 pp en recall, -2 pp en precisión

**Flujo de Ejecución:**
1. **Cargar modelo entrenado**
2. **Procesar imágenes por año (2020-2025)**
3. **Para cada tesela:**
   - Predicción
   - Post-procesamiento morfológico
   - Guardado de máscara individual
4. **Mosaicado por año:** Fusión de todas las máscaras
5. **Validación:** Comparación con máscaras anotadas (si disponibles)
6. **Reporte:** Generación de métricas y visualizaciones

**Uso:**
```bash
# Desde directorio satseg-main (por compatibilidad de rutas)
cd /ruta/a/satseg-main
uv run predict_and_mosaic_with_metrics.py
```

**Salida:**
```
predicciones_por_año/
├── 2020/
│   ├── mosaico_2020.tif      # Mosaico completo del año
│   ├── mask_*.tif            # Máscaras individuales
│   └── metrics_2020.json     # Métricas de validación
├── 2021/
│   └── ...
├── matrices_confusion/       # Matrices de confusión por año
└── reporte_general.txt       # Resumen consolidado
```

---

## 🔍 Especificación Técnica de Datos

### Características de Imágenes (Tiles)
- **Fuente:** Sentinel-2 (ESA)
- **Bandas:** 11 (Blue, Green, Red, NIR, SWIR1, SWIR2, etc.)
- **Resolución:** 10-20m según banda
- **Tamaño:** 256×256 píxeles (~2.56 km² a 5.12 km²)
- **Rango de valores:** [0, 1] normalizado
- **Proyección:** UTM 17S (EPSG:32717)
- **Temporalidad:** Anual (2020-2025)

### Características de Máscaras
- **Tipo:** Binarias (1-canal)
- **Valores:** 0 (no-manglar), 1 (manglar)
- **Anotación:** Manual por expertos
- **Tamaño:** Coincide con tiles (256×256)

---

## 🚀 Flujo de Trabajo Típico

### 1️⃣ Entrenamiento
```bash
# Entrenar modelo Multi-Branch UNet++
python model/train_multibranch_v_copy_v1.py
# Genera: checkpoints/best_model.pth
```

### 2️⃣ Inferencia y Mosaicado
```bash
# Predecir sobre datos de prueba y generar mosaicos
python Script/predict_and_mosaic_with_metrics.py
# Genera: predicciones_por_año/{año}/mosaico_{año}.tif
```

### 3️⃣ Evaluación
```bash
# En Jupyter, analizar resultados
jupyter notebook Script/train_datos_refactored.ipynb
```

---

## 📊 Estadísticas del Dataset

### Distribución de Clases (TRAIN)
| Categoría | Cantidad | Porcentaje | Rango Manglar |
|-----------|----------|-----------|---------------|
| Negative | 15 | 4.6% | 0% |
| Positive Dense | 276 | 85.5% | >20% |
| Positive Sparse | 32 | 10% | 0-20% |
| **TOTAL** | **323** | **100%** | - |

### Ejemplos de Pares Imagen-Máscara
- `tile_0000.tif / mask_0000.tif` → 9.74% manglar (positive_dense)
- `tile_0001.tif / mask_0001.tif` → 0.00% manglar (negative)
- `tile_0002.tif / mask_0002.tif` → 47.66% manglar (positive_dense)

---

## 🔧 Requisitos y Dependencias

**Python >= 3.10**

### Librerías Principales
- `torch` - Framework de deep learning
- `segmentation_models_pytorch` - Modelos preentrenados
- `rasterio` - Lectura/escritura de datos geoespaciales
- `geopandas` - Manipulación de geometrías (shapefiles)
- `torchgeo` - Extensión de PyTorch para datos geoespaciales
- `scipy` - Procesamiento morfológico
- `matplotlib`, `seaborn` - Visualización
- `tqdm` - Barras de progreso

**Instalación:**
```bash
pip install torch torchgeo rasterio geopandas scipy matplotlib seaborn segmentation-models-pytorch
```

---

## 📈 Métricas Principales

El proyecto calcula:
- **IoU (Intersection over Union):** Métrica estándar de segmentación
- **Precisión:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** Media armónica de precisión y recall
- **Matriz de Confusión:** TP, TN, FP, FN

---

## 🎨 Visualizaciones Generadas

1. **Curvas de entrenamiento:** Pérdida y métrica por época
2. **Distribución de clases:** Histogramas train/val
3. **Mosaicos predichos:** Imágenes RGB + máscaras predichas
4. **Matrices de confusión:** Por año y consolidada
5. **Overlays:** Comparación máscara real vs predicha

---

## 📝 Notas Importantes

### Optimización de Normalización (Diciembre 2024)
Se identificó que las imágenes Sentinel-2 de Google Earth Engine **ya están normalizadas [0,1]**. Anteriormente se aplicaba una normalización percentil redundante que comprimía el rango dinámico. Esto fue corregido en `predict_and_mosaic_with_metrics.py` con:
- ✅ Eliminación de normalización percentil
- ✅ Threshold optimizado a 0.50
- ✅ Ganancia: +8-10 pp en recall

### Rutas de Ejecución
El script `predict_and_mosaic_with_metrics.py` debe ejecutarse desde `/satseg-main` debido a la estructura de rutas relativas de módulos internos.

---

## 📚 Referencias Bibliográficas

1. **Multi-Branch UNet++:**
   - Zhou et al. (2018): "UNet++: A Nested U-Net Architecture"
   - Cao et al. (2021): "Dual Stream Fusion Network for Multi-spectral HRRS"

2. **DeepLabV3+:**
   - Chen et al. (2017): "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets"

3. **SegFormer:**
   - Xie et al. (2021): "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"

4. **Datasets Sentinel-2:**
   - ESA Copernicus: https://www.copernicus.eu/

---

## 👤 Autoría

Proyecto: SatSem  
Área de Estudio: Jambeli, Ecuador  
Aplicación: Segmentación de Manglares en Teledetección  
Publicación: Remote Sensing Applications Society and Environment (RSASE)

---

**Última actualización:** 30 de enero de 2026
