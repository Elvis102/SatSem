"""
Script de entrenamiento unificado para segmentación semántica con TorchGeo.

Soporta múltiples arquitecturas para segmentación de manglares con Sentinel-2:

═══════════════════════════════════════════════════════════════════════════════
MODELOS CNN (Deep Learning):
═══════════════════════════════════════════════════════════════════════════════

1. Multi-Branch UNet++ (default)
   - Procesamiento multi-resolución: High-Res (10m) + Low-Res (20m)
   - Fusión FPN-style con nested skip connections
   - Deep supervision opcional

2. UNet / UNet++ / DeepLabV3+ / PSPNet / HRNet
   - Modelos estándar de SMP (segmentation_models_pytorch)
   - Encoders: ResNet18/34/50/101, EfficientNet, InceptionResNetV2, HRNet

Basado en:
- Cao et al. (2021): "Dual Stream Fusion Network for Multi-spectral HRRS"
- Zhou et al. (2018): "UNet++: A Nested U-Net Architecture"
- Chen et al. (2017): "DeepLab: Semantic Image Segmentation"
- Zhao et al. (2017): "PSPNet: Pyramid Scene Parsing Network"

═══════════════════════════════════════════════════════════════════════════════
MODELO TRANSFORMER (Vision Transformer):
═══════════════════════════════════════════════════════════════════════════════

3. SegFormer (Hugging Face Transformers)
   - Arquitectura de Transformer eficiente para segmentación semántica
   - Encoder ViT jerárquico con atención eficiente
   - Decoder MLP All-Layer ligero
   - Sin embeddings posicionales (flexible a diferentes resoluciones)
   - Variantes: B0 (ligero) a B5 (pesado)

   DIFERENCIAS METODOLÓGICAS CON CNNs:
   ┌─────────────────────────────────────────────────────────────────────┐
   │                 SegFormer                │         CNN (UNet++)      │
   ├──────────────────────────────────────────┼───────────────────────────┤
   │ Atención global desde el inicio          │ Receptive field local     │
   │ Mejor para contexto a larga distancia    │ Mejor para texturas       │
   │ Sin embeddings posicionales (flexible)   │ Tamaño fijo implícito     │
   │ Decoder MLP ligero                       │ Decoder convolucional     │
   │ Menos parámetros que ViT estándar        │ Más parámetros típicamente│
   └──────────────────────────────────────────┴───────────────────────────┘

   VENTAJAS PARA MANGLAR DISPERSO:
   - Mejor captura de contexto global (importante para parches dispersos)
   - Atención multi-escala sin pooling agresivo
   - Potencialmente mejor IoU en teselas positive_sparse

   USO:
   1. Cambiar MODEL_TYPE = "segformer" en Config
   2. Configurar SEGFORMER_VARIANT ('b0', 'b1', etc.)
   3. Ajustar LR (típicamente 6e-5 a 2e-4)
   4. Ejecutar script normalmente

   Basado en:
   - Xie et al. (2021): "SegFormer: Simple and Efficient Design for Semantic
     Segmentation with Transformers"

═══════════════════════════════════════════════════════════════════════════════
MODELO BASELINE (Machine Learning Clásico):
═══════════════════════════════════════════════════════════════════════════════

4. Random Forest Classifier
   - Clasificador ensemble de árboles de decisión
   - Clasificación píxel a píxel con features diseñadas manualmente
   - Features: bandas espectrales + índices (NDVI, NDWI, etc.) + contexto espacial

   DIFERENCIAS METODOLÓGICAS CON CNNs:
   ┌─────────────────────────────────────────────────────────────────────┐
   │                    CNN                 │     Random Forest           │
   ├────────────────────────────────────────┼─────────────────────────────┤
   │ Features automáticas                   │ Features manuales           │
   │ Contexto espacial grande               │ Contexto limitado (ventana) │
   │ Visión holística de imagen             │ Clasificación píxel a píxel │
   │ Requiere GPU + mucho tiempo            │ CPU-only + rápido           │
   │ Difícil interpretación                 │ Feature importance          │
   │ Necesita data augmentation             │ Robusto sin augmentation    │
   └────────────────────────────────────────┴─────────────────────────────┘

   LIMITACIONES CONCEPTUALES:
   - RF no aprende representaciones jerárquicas como CNNs
   - Depende de calidad de índices espectrales diseñados
   - Ventana espacial fija vs receptive field adaptativo de CNNs
   - Comparación no totalmente justa, pero útil como baseline

   USO:
   1. Cambiar MODEL_TYPE = "random_forest" en Config
   2. Configurar RF_WINDOW_SIZE, RF_USE_INDICES, RF_N_ESTIMATORS
   3. Ejecutar script normalmente
   4. Comparar métricas en outputs/stats/

═══════════════════════════════════════════════════════════════════════════════

Características generales:
1. Compatible con TorchGeo DataModule
2. Análisis automático de distribución de clases (train + val)
3. Generación de figuras de alta calidad (300 DPI)
4. Métricas estandarizadas: IoU, F1, Kappa, Accuracy, Precision, Recall, FWIoU
5. Métricas IoU por clase (manglar / no-manglar) para análisis detallado
6. Guardado automático de resultados para comparación cuantitativa

Bandas Sentinel-2:
- High-Res (10m): B2, B3, B4, B8 (Blue, Green, Red, NIR)
- Low-Res (20m): B11, B12 (SWIR1, SWIR2)
"""

import torch
import lightning as L
import warnings
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Imports locales
from src.dm_torchgeo import TorchGeoDataModule
from src.module import Module
from src.metrics import iou, f1_score, overall_accuracy, precision, recall, fw_iou, kappa
from src.models.multi_branch_unet import MultiBranchUNetWrapper
from src.metrics_logger import MetricsHistoryCallback

# SegFormer (Hugging Face Transformers)
from src.models.segformer_wrapper import (
    SegFormerWrapper,
    SegFormerWithAuxLoss,
    create_segformer_model
)

# SegNet (Arquitectura clásica encoder-decoder)
from src.models.segnet import (
    SegNet,
    SegNetLite,
    create_segnet_model
)

# SegNeXt (CNN moderna con atención multi-escala) - SOTA eficiente
from src.models.segnext import (
    SegNeXt,
    create_segnext_model
)

# Mask2Former (Transformer SOTA) - Requiere más recursos
from src.models.mask2former_wrapper import (
    Mask2FormerWrapper,
    create_mask2former_model
)

# Swin Transformer V2 + UPerNet (Transformer jerárquico SOTA)
from src.models.swin_upernet import (
    SwinV2UPerNetWrapper,
    SwinV2UNet,
    create_swinv2_model
)

# Lightning
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback

# Transformaciones
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K

# Segmentation Models
import segmentation_models_pytorch as smp

# Scikit-learn (para Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import json

# Configurar warnings
warnings.filterwarnings('ignore', message='.*align_corners.*')
warnings.filterwarnings('ignore', message='.*grid_sample.*')
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración centralizada del experimento."""
    
    # --- Datos ---
    DATA_DIR = '/home/elvissr/thesis_project/data/processed'
    BATCH_SIZE = 32
    NUM_WORKERS = 64
    CACHE_DATA = True
    TILE_SIZE = 128  # Tamaño de las teselas (128x128 píxeles)
    
    # Configuración de bandas Multi-Resolución
    # IMPORTANTE: El orden de bandas debe ser [High-Res, Low-Res]
    # High-Res (10m): B2, B3, B4, B8
    # Low-Res (20m): B11, B12
    # Banda 0: B2, Banda 1: B3, Banda 2: B4, Banda 3: B8, Banda 4: B11, Banda 5: B12
    BANDS = [0, 1, 2, 3, 4, 5]  # Todas las bandas disponibles
    HIGH_RES_CHANNELS = 4  # B2, B3, B4, B8
    LOW_RES_CHANNELS = 2   # B11, B12
    TOTAL_CHANNELS = HIGH_RES_CHANNELS + LOW_RES_CHANNELS  # 6 bandas totales
    NORM_VALUE = 1.0
    
    # --- Modelo ---
    # MODEL_TYPE opciones:
    #   CNN CLÁSICAS:
    #   - 'multi_branch': UNet++ Multi-Branch con fusión multi-resolución (default)
    #   - 'unetplusplus': UNet++ estándar (nested skip connections)
    #   - 'unet': UNet básico estándar
    #   - 'unet_inception': UNet con encoder InceptionResNetV2
    #   - 'deeplabv3plus': DeepLabV3+ con ASPP
    #   - 'hrnet': HRNet (High-Resolution Net) con representaciones multi-escala
    #   - 'pspnet': PSPNet (Pyramid Scene Parsing Network) con pooling piramidal
    #   - 'segnet': SegNet (Arquitectura clásica encoder-decoder)
    #
    #   TRANSFORMERS:
    #   - 'segformer': SegFormer (Vision Transformer eficiente)
    #   - 'mask2former': Mask2Former (SOTA universal, más pesado)
    #   - 'swinv2': Swin Transformer V2 + UPerNet (SOTA jerárquico)
    #
    #   CNN MODERNAS (SOTA):
    #   - 'segnext': SegNeXt (CNN con atención multi-escala, muy eficiente) ← RECOMENDADO
    #
    #   BASELINE ML:
    #   - 'random_forest': Random Forest Classifier (baseline clásico de ML)
    MODEL_TYPE = "multi_branch"  # ← CAMBIAR PARA PROBAR OTROS MODELOS

    # ENCODER: depende del MODEL_TYPE
    #   - Para multi_branch, unet, unetplusplus, deeplabv3plus, pspnet: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'efficientnet-b0', 'hrnet_w18', 'hrnet_w32', 'hrnet_w40', 'hrnet_w48'
    #   - Para unet_inception: se usa automáticamente 'inceptionresnetv2'
    #   - Para hrnet: 'hrnet_w18', 'hrnet_w32', 'hrnet_w40', 'hrnet_w48' (w48 recomendado)
    #   - Para pspnet: 'resnet50', 'resnet101' (resnet101 recomendado para mejor contexto)
    #   - Para random_forest: N/A (no usa encoder)
    ENCODER = "resnet101"
    ENCODER_WEIGHTS = "imagenet"  # 'imagenet', None
    FUSION_MODE = "fpn"  # 'fpn', 'concat', 'add' (solo para multi_branch)
    UPSAMPLE_MODE = "bilinear"  # 'bilinear', 'bicubic', 'nearest'
    DEEP_SUPERVISION = False  # Desactivado: SMP UNet++ decoder no expone features intermedias
    NUM_CLASSES = 1  # Segmentación binaria

    # --- Configuración Random Forest (solo si MODEL_TYPE = 'random_forest') ---
    RF_N_ESTIMATORS = 200          # Número de árboles en el bosque
    RF_MAX_DEPTH = 30              # Profundidad máxima de cada árbol
    RF_MIN_SAMPLES_SPLIT = 10      # Mínimo de muestras para dividir un nodo
    RF_MIN_SAMPLES_LEAF = 5        # Mínimo de muestras en una hoja
    RF_WINDOW_SIZE = 1             # Tamaño de ventana de contexto espacial (5x5 recomendado)
    RF_USE_INDICES = False          # Calcular índices espectrales (NDVI, NDWI, etc.)
    RF_MAX_SAMPLES = 2_000_000     # Máximo de píxeles para entrenamiento (submuestreo)
    RF_N_JOBS = -1                 # Número de cores (-1 = todos disponibles)
    RF_RANDOM_STATE = 42           # Semilla para reproducibilidad

    # --- Configuración SegFormer (solo si MODEL_TYPE = 'segformer') --- NUEVO
    # VARIANTES DISPONIBLES:
    #   'b0': ~3.7M params - Más ligero, recomendado para teselas 128x128
    #   'b1': ~13.7M params - Balance rendimiento/eficiencia
    #   'b2': ~27.4M params - Mayor capacidad
    #   'b3': ~47.3M params - Para datasets grandes
    #   'b4': ~64.1M params - Alta capacidad
    #   'b5': ~84.7M params - Máxima capacidad
    SEGFORMER_VARIANT = 'b3'       # Variante del modelo (b0-b5)
    SEGFORMER_PRETRAINED = True    # Usar pesos preentrenados de ImageNet
    SEGFORMER_USE_AUX_LOSS = False # Cabeza auxiliar para deep supervision
    SEGFORMER_AUX_WEIGHT = 0.4     # Peso de la pérdida auxiliar (si USE_AUX_LOSS=True)

    # RECOMENDACIONES DE HIPERPARÁMETROS PARA SEGFORMER:
    # ┌────────────────────────────────────────────────────────────────────────┐
    # │ Parámetro        │ Valor recomendado │ Notas                           │
    # ├──────────────────┼───────────────────┼─────────────────────────────────┤
    # │ Learning Rate    │ 6e-5 a 2e-4       │ Transformers necesitan LR menor │
    # │ Batch Size       │ 32 (B0), 16 (B1)  │ Según memoria GPU disponible    │
    # │ Weight Decay     │ 0.01              │ Mayor que CNNs típicamente      │
    # │ Epochs           │ 100-200           │ Similar a CNNs                  │
    # │ Warmup           │ 10% de épocas     │ Opcional pero recomendado       │
    # │ Scheduler        │ ReduceLROnPlateau │ O polynomial decay              │
    # └──────────────────┴───────────────────┴─────────────────────────────────┘

    # --- Configuración SegNet (solo si MODEL_TYPE = 'segnet') ---
    # VARIANTES DISPONIBLES:
    #   'standard': SegNet completo (VGG16-style, ~29M params)
    #   'lite': Versión ligera para imágenes pequeñas (~7M params)
    SEGNET_VARIANT = 'lite'  # 'standard' o 'lite' (recomendado para 128x128)

    # NOTA SOBRE SEGNET:
    # SegNet es una arquitectura clásica (2017) incluida para comparación histórica.
    # Típicamente tiene menor rendimiento que UNet debido a la ausencia de skip
    # connections. Se espera IoU 5-10% menor que UNet en la mayoría de casos.

    # --- Configuración SegNeXt (solo si MODEL_TYPE = 'segnext') --- RECOMENDADO
    # VARIANTES DISPONIBLES:
    #   'tiny': ~4M params - Muy eficiente, ideal para 128x128
    #   'small': ~14M params - Buen balance rendimiento/velocidad
    #   'base': ~28M params - Mayor capacidad
    #   'large': ~49M params - Máxima capacidad
    SEGNEXT_VARIANT = 'base'  # 'tiny', 'small', 'base', 'large'

    # NOTA SOBRE SEGNEXT:
    # SegNeXt es SOTA en eficiencia para segmentación semántica (NeurIPS 2022).
    # Supera a SegFormer con menos FLOPs usando atención convolucional multi-escala.
    # Recomendado como primera opción para comparación con CNNs tradicionales.

    # --- Configuración Mask2Former (solo si MODEL_TYPE = 'mask2former') ---
    # VARIANTES DISPONIBLES:
    #   'tiny': ~47M params - Recomendado para 128x128
    #   'small': ~69M params - Mayor capacidad
    #   'base': ~102M params - Para datasets grandes
    #   'large': ~216M params - Máxima capacidad
    MASK2FORMER_VARIANT = 'tiny'  # 'tiny', 'small', 'base', 'large'

    # NOTA SOBRE MASK2FORMER:
    # Mask2Former es SOTA absoluto en segmentación universal (NeurIPS 2022).
    # Usa masked cross-attention para mejor convergencia.
    # Requiere más recursos (GPU/memoria) que otros modelos.
    # Usar solo si se dispone de GPU con >8GB VRAM.

    # --- Configuración Swin V2 (solo si MODEL_TYPE = 'swinv2') ---
    # VARIANTES DISPONIBLES:
    #   'tiny': ~60M params - Recomendado para 128x128
    #   'small': ~81M params - Mayor capacidad
    #   'base': ~121M params - Para datasets grandes
    #   'large': ~234M params - Máxima capacidad
    SWINV2_VARIANT = 'tiny'  # 'tiny', 'small', 'base', 'large'

    # DECODER:
    #   'upernet': UPerNet decoder (SOTA, más parámetros)
    #   'unet': UNet-style decoder (más ligero, skip connections)
    SWINV2_DECODER = 'unet'  # 'upernet', 'unet'

    # NOTA SOBRE SWIN V2:
    # Swin Transformer V2 mejora sobre V1 con:
    # - Log-spaced continuous position bias (mejor para diferentes resoluciones)
    # - Scaled cosine attention (mejor estabilidad de entrenamiento)
    # - Residual post normalization (mejor convergencia)
    # Combinado con UPerNet para feature fusion multi-escala.

    # --- Entrenamiento ---
    # Learning rate base (para CNNs)
    LR_BASE = 1e-4

    # Learning rates específicos por tipo de modelo
    # Los Transformers que entrenan desde cero necesitan LR más bajo
    LR_BY_MODEL = {
        'swinv2': 1e-5,       # Swin V2 entrena desde cero - necesita LR MUY bajo
        'mask2former': 5e-5,  # Mask2Former - Transformer complejo
        'segformer': 1e-4,    # SegFormer usa backbone preentrenado
        'segnext': 1e-4,      # CNN moderna
    }

    # Seleccionar LR según modelo
    LR = LR_BY_MODEL.get(MODEL_TYPE, LR_BASE)

    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PRECISION = '16-mixed'

    # Warmup para Transformers (número de épocas con LR gradual)
    WARMUP_EPOCHS_BY_MODEL = {
        'swinv2': 10,         # 10 épocas de warmup para Swin V2
        'mask2former': 5,     # 5 épocas para Mask2Former
    }
    WARMUP_EPOCHS = WARMUP_EPOCHS_BY_MODEL.get(MODEL_TYPE, 0)

    # --- Función de Pérdida y Manejo de Desbalance ---
    # Flag principal: True = compensación explícita, False = configuración estándar
    USE_CLASS_BALANCING = True  # ← CAMBIAR A False PARA DESACTIVAR COMPENSACIÓN

    # Distribución de clases en entrenamiento: 22.70% manglar, 77.30% no-manglar
    # Configuración CON compensación de desbalance (USE_CLASS_BALANCING = True):
    FOCAL_ALPHA = 0.69         # Compensación explícita del desbalance (ratio ~1:3.4)
    FOCAL_GAMMA = 2.0           # Factor de down-weight para ejemplos fáciles
    DICE_SMOOTH = 1.0           # Suavizado para estabilidad numérica en Dice Loss
    LOSS_WEIGHT_DICE = 0.6      # Peso de Dice Loss en la combinación
    LOSS_WEIGHT_FOCAL = 0.4     # Peso de Focal Loss en la combinación

    # Configuración SIN compensación de desbalance (USE_CLASS_BALANCING = False):
    # - Focal Loss: alpha=None (sin ponderación de clases), gamma=2.0
    # - Dice Loss: smooth=1.0
    # - Pesos: 1.0·Dice + 1.0·Focal (suma simple sin ponderación)
    
    # --- Scheduler ---
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MONITOR = 'val_iou'
    
    # --- Callbacks ---
    EARLY_STOPPING_PATIENCE = 30
    CHECKPOINT_MONITOR = 'val_iou'
    
    # --- Logging ---
    EXPERIMENT_NAME = 'satseg-multibranch'
    # RUN_NAME se genera dinámicamente según MODEL_TYPE
    if MODEL_TYPE == "deeplabv3plus":
        RUN_NAME = f'DeepLabV3Plus-{ENCODER}-Sentinel2'
    elif MODEL_TYPE == "unet":
        RUN_NAME = f'UNet-{ENCODER}-Sentinel2'
    elif MODEL_TYPE == "unet_inception":
        RUN_NAME = f'UNet-InceptionResNetV2-Sentinel2'
    elif MODEL_TYPE == "unetplusplus":
        RUN_NAME = f'UNetPlusPlus-{ENCODER}-Sentinel2'
    elif MODEL_TYPE == "hrnet":
        RUN_NAME = f'HRNet-{ENCODER}-Sentinel2'
    elif MODEL_TYPE == "pspnet":
        RUN_NAME = f'PSPNet-{ENCODER}-Sentinel2'
    elif MODEL_TYPE == "segformer":
        aux_str = '-aux' if SEGFORMER_USE_AUX_LOSS else ''
        RUN_NAME = f'SegFormer-{SEGFORMER_VARIANT.upper()}{aux_str}-Sentinel2'
    elif MODEL_TYPE == "segnet":
        RUN_NAME = f'SegNet-{SEGNET_VARIANT.capitalize()}-Sentinel2'
    elif MODEL_TYPE == "segnext":
        RUN_NAME = f'SegNeXt-{SEGNEXT_VARIANT.capitalize()}-Sentinel2'
    elif MODEL_TYPE == "mask2former":
        RUN_NAME = f'Mask2Former-{MASK2FORMER_VARIANT.capitalize()}-Sentinel2'
    elif MODEL_TYPE == "swinv2":
        RUN_NAME = f'SwinV2-{SWINV2_VARIANT.capitalize()}-{SWINV2_DECODER.capitalize()}-Sentinel2'
    elif MODEL_TYPE == "random_forest":
        window_str = f'w{RF_WINDOW_SIZE}x{RF_WINDOW_SIZE}' if RF_WINDOW_SIZE > 1 else 'nowin'
        indices_str = 'idx' if RF_USE_INDICES else 'noidx'
        RUN_NAME = f'RandomForest-{window_str}-{indices_str}-{RF_N_ESTIMATORS}trees-Sentinel2'
    else:  # multi_branch
        RUN_NAME = f'MultiBranch-UNetPP-{ENCODER}-{FUSION_MODE}-Sentinel2'
    LOG_EVERY_N_STEPS = 10
    
    # --- Directorios ---
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    OUTPUT_DIR = 'outputs'
    FIGURES_DIR = 'outputs/figures'
    STATS_DIR = 'outputs/stats'


def print_config():
    """Imprime la configuración del experimento."""
    print("\n" + "="*80)
    model_names = {
        "multi_branch": "MULTI-BRANCH UNET++",
        "unetplusplus": "UNET++ ESTÁNDAR",
        "unet": "UNET BÁSICO",
        "unet_inception": "UNET + INCEPTIONRESNETV2",
        "deeplabv3plus": "DeepLabV3+",
        "hrnet": "HRNET (HIGH-RESOLUTION NET)",
        "pspnet": "PSPNET (PYRAMID SCENE PARSING NETWORK)",
        "segformer": "SEGFORMER (VISION TRANSFORMER)",
        "segnet": "SEGNET (ENCODER-DECODER CLÁSICO)",
        "segnext": "SEGNEXT (CNN MODERNA SOTA)",
        "mask2former": "MASK2FORMER (TRANSFORMER SOTA)",
        "swinv2": "SWIN TRANSFORMER V2 + UPERNET",
        "random_forest": "RANDOM FOREST CLASSIFIER (BASELINE ML)"
    }
    print(f"CONFIGURACIÓN {model_names.get(Config.MODEL_TYPE, Config.MODEL_TYPE.upper())}")
    print("="*80)
    print(f"🧠 Modelo: {Config.MODEL_TYPE.upper()}")
    print(f"📊 Bandas High-Res (10m): {Config.HIGH_RES_CHANNELS} (RGB + NIR)")
    print(f"📊 Bandas Low-Res (20m): {Config.LOW_RES_CHANNELS} (SWIR1 + SWIR2)")
    print(f"📊 Total de bandas: {Config.TOTAL_CHANNELS}")

    # Mostrar encoder según el modelo
    if Config.MODEL_TYPE == "unet_inception":
        print(f"🏗️  Encoder: inceptionresnetv2 (fijo)")
    else:
        print(f"🏗️  Encoder: {Config.ENCODER}")

    # Mostrar decoder según el modelo
    if Config.MODEL_TYPE == "deeplabv3plus":
        print(f"🔧 Decoder: DeepLabV3+ (ASPP + Decoder)")
        print(f"🔗 Output Stride: 16")
        print(f"⬆️  ASPP Rates: (12, 24, 36)")
    elif Config.MODEL_TYPE == "unet":
        print(f"🔧 Decoder: UNet (Skip Connections)")
        print(f"⬆️  Decoder Channels: (256, 128, 64, 32, 16)")
    elif Config.MODEL_TYPE == "unet_inception":
        print(f"🔧 Decoder: UNet (Skip Connections)")
        print(f"🔗 Encoder: InceptionResNetV2 (Google)")
        print(f"⬆️  Decoder Channels: (256, 128, 64, 32, 16)")
    elif Config.MODEL_TYPE == "unetplusplus":
        print(f"🔧 Decoder: UNet++ (Nested Skip Connections)")
        print(f"⬆️  Decoder Channels: (256, 128, 64, 32, 16)")
        print(f"📝 Nota: NO usa Multi-Branch (bandas concatenadas)")
    elif Config.MODEL_TYPE == "hrnet":
        print(f"🔧 Decoder: Segmentation Head")
        print(f"🏗️  Arquitectura: Representaciones multi-escala paralelas")
        print(f"🔗 Característica principal: Mantiene alta resolución durante toda la red")
        print(f"📝 Nota: HRNet intercambia información entre resoluciones múltiples")
    elif Config.MODEL_TYPE == "pspnet":
        print(f"🔧 Decoder: Pyramid Pooling Module (PPM)")
        print(f"🏗️  Arquitectura: Pooling piramidal multi-escala")
        print(f"🔗 PSP Sizes: (1, 2, 3, 6) - pooling a diferentes escalas")
        print(f"📝 Nota: PSPNet agrega contexto global mediante pyramid pooling")
    elif Config.MODEL_TYPE == "segformer":
        # Información de parámetros aproximados por variante
        params_by_variant = {
            'b0': '~3.7M', 'b1': '~13.7M', 'b2': '~27.4M',
            'b3': '~47.3M', 'b4': '~64.1M', 'b5': '~84.7M'
        }
        variant_params = params_by_variant.get(Config.SEGFORMER_VARIANT, 'N/A')
        print(f"🤖 Arquitectura: Vision Transformer (ViT jerárquico)")
        print(f"🏗️  Variante: SegFormer-{Config.SEGFORMER_VARIANT.upper()}")
        print(f"📊 Parámetros aproximados: {variant_params}")
        print(f"🔗 Encoder: ViT jerárquico con atención eficiente")
        print(f"🔧 Decoder: MLP All-Layer (fusión multi-escala)")
        print(f"📥 Pesos preentrenados: {'✅ ImageNet' if Config.SEGFORMER_PRETRAINED else '❌ Desde cero'}")
        if Config.SEGFORMER_USE_AUX_LOSS:
            print(f"🎓 Deep Supervision: ✅ ACTIVADA (peso aux: {Config.SEGFORMER_AUX_WEIGHT})")
        else:
            print(f"🎓 Deep Supervision: ❌ Desactivada")
        print(f"📝 Nota: SegFormer usa atención global, ideal para manglar disperso")
        print(f"📝 Referencia: Xie et al. (2021) - SegFormer")
    elif Config.MODEL_TYPE == "segnet":
        params_by_variant = {'standard': '~29M', 'lite': '~7M'}
        variant_params = params_by_variant.get(Config.SEGNET_VARIANT, 'N/A')
        print(f"🏛️  Arquitectura: Encoder-Decoder clásico (VGG16-style)")
        print(f"🏗️  Variante: SegNet-{Config.SEGNET_VARIANT.capitalize()}")
        print(f"📊 Parámetros aproximados: {variant_params}")
        print(f"🔗 Encoder: 5 bloques convolucionales con max pooling")
        print(f"🔧 Decoder: Unpooling con índices + convoluciones")
        print(f"📝 Característica: Sin skip connections (a diferencia de UNet)")
        print(f"📝 Nota: Arquitectura clásica (2017) - IoU típicamente menor que UNet")
        print(f"📝 Referencia: Badrinarayanan et al. (2017) - SegNet")
    elif Config.MODEL_TYPE == "segnext":
        params_by_variant = {'tiny': '~4M', 'small': '~14M', 'base': '~28M', 'large': '~49M'}
        variant_params = params_by_variant.get(Config.SEGNEXT_VARIANT, 'N/A')
        print(f"🚀 Arquitectura: CNN con atención convolucional multi-escala (SOTA)")
        print(f"🏗️  Variante: SegNeXt-{Config.SEGNEXT_VARIANT.capitalize()}")
        print(f"📊 Parámetros aproximados: {variant_params}")
        print(f"🔗 Encoder: MSCAN (Multi-Scale Convolutional Attention Network)")
        print(f"🔧 Decoder: Hamburger (fusión global eficiente)")
        print(f"📝 Característica: Strip convolutions multi-escala")
        print(f"📝 Ventaja: Supera SegFormer con menos FLOPs")
        print(f"📝 Referencia: Guo et al. (2022) - SegNeXt (NeurIPS)")
    elif Config.MODEL_TYPE == "mask2former":
        params_by_variant = {'tiny': '~47M', 'small': '~69M', 'base': '~102M', 'large': '~216M'}
        variant_params = params_by_variant.get(Config.MASK2FORMER_VARIANT, 'N/A')
        print(f"🏆 Arquitectura: Transformer SOTA con Masked Attention")
        print(f"🏗️  Variante: Mask2Former-{Config.MASK2FORMER_VARIANT.capitalize()}")
        print(f"📊 Parámetros aproximados: {variant_params}")
        print(f"🔗 Backbone: Swin Transformer")
        print(f"🔧 Decoder: Pixel decoder + Transformer decoder")
        print(f"📝 Característica: Masked cross-attention para queries")
        print(f"📝 Nota: SOTA absoluto - requiere más recursos (GPU >8GB)")
        print(f"📝 Referencia: Cheng et al. (2022) - Mask2Former (NeurIPS)")
    elif Config.MODEL_TYPE == "swinv2":
        params_by_variant = {'tiny': '~60M', 'small': '~81M', 'base': '~121M', 'large': '~234M'}
        variant_params = params_by_variant.get(Config.SWINV2_VARIANT, 'N/A')
        print(f"🏆 Arquitectura: Swin Transformer V2 + {Config.SWINV2_DECODER.upper()}")
        print(f"🏗️  Variante: SwinV2-{Config.SWINV2_VARIANT.capitalize()}")
        print(f"📊 Parámetros aproximados: {variant_params}")
        print(f"🔗 Backbone: Swin Transformer V2 (shifted windows)")
        print(f"🔧 Decoder: {Config.SWINV2_DECODER.upper()}")
        print(f"📝 Característica: Log-spaced continuous position bias")
        print(f"📝 Mejoras V2: Scaled cosine attention + Residual post-norm")
        print(f"📝 Referencia: Liu et al. (2022) - Swin Transformer V2 (CVPR)")
    elif Config.MODEL_TYPE == "random_forest":
        print(f"🌲 Clasificador: Random Forest (Ensemble de árboles de decisión)")
        print(f"🔢 N° de árboles: {Config.RF_N_ESTIMATORS}")
        print(f"📏 Profundidad máxima: {Config.RF_MAX_DEPTH}")
        print(f"🪟 Ventana de contexto: {Config.RF_WINDOW_SIZE}x{Config.RF_WINDOW_SIZE}")
        print(f"📐 Índices espectrales: {'✅ Activados' if Config.RF_USE_INDICES else '❌ Desactivados'}")
        if Config.RF_USE_INDICES:
            print(f"   ├─ NDVI (Vegetation Index)")
            print(f"   ├─ NDWI (Water Index)")
            print(f"   ├─ EVI (Enhanced Vegetation)")
            print(f"   ├─ SAVI (Soil Adjusted)")
            print(f"   ├─ MNDWI (Modified Water)")
            print(f"   └─ NIR/Red Ratio")
        print(f"📊 Max samples training: {Config.RF_MAX_SAMPLES:,}")
        print(f"⚙️  N° cores: {Config.RF_N_JOBS} (todos)")
        print(f"📝 Nota: Clasificación píxel a píxel con features diseñadas manualmente")
    else:  # multi_branch
        print(f"🔧 Decoder: UNet++ (Nested Skip Connections)")
        print(f"🔗 Fusion Mode: {Config.FUSION_MODE}")
        print(f"⬆️  Upsample Mode: {Config.UPSAMPLE_MODE}")
        print(f"📝 Nota: SÍ usa Multi-Branch (10m y 20m procesados por separado)")

    # Deep Supervision con información detallada
    if Config.DEEP_SUPERVISION:
        print(f"🎓 Deep Supervision: ✅ ACTIVADA")
        print(f"   ├─ Outputs: 4 escalas (16, 32, 64, 128 canales)")
        print(f"   ├─ Pérdida: Ponderada [0.5, 0.25, 0.15, 0.1]")
        print(f"   └─ Métricas: Calculadas sobre output final")
    else:
        print(f"🎓 Deep Supervision: ❌ Desactivada")

    print(f"📦 Batch Size: {Config.BATCH_SIZE}")
    lr_note = f" (específico para {Config.MODEL_TYPE})" if Config.MODEL_TYPE in Config.LR_BY_MODEL else " (default)"
    print(f"🎯 Learning Rate: {Config.LR}{lr_note}")
    if Config.WARMUP_EPOCHS > 0:
        print(f"⚡ Warmup: {Config.WARMUP_EPOCHS} épocas (LR: {Config.LR*0.1:.1e} → {Config.LR:.1e})")
    print(f"🔄 Epochs: {Config.EPOCHS}")
    print(f"💾 Cache Data: {Config.CACHE_DATA}")

    # Configuración de pérdida y manejo de desbalance
    print(f"\n⚖️  MANEJO DE DESBALANCE DE CLASES:")
    if Config.USE_CLASS_BALANCING:
        print(f"   ├─ Modo: ✅ COMPENSACIÓN EXPLÍCITA ACTIVADA")
        print(f"   ├─ Focal Loss α: {Config.FOCAL_ALPHA} (compensación clase minoritaria)")
        print(f"   ├─ Focal Loss γ: {Config.FOCAL_GAMMA} (down-weight ejemplos fáciles)")
        print(f"   ├─ Dice smooth: {Config.DICE_SMOOTH}")
        print(f"   └─ Pesos: {Config.LOSS_WEIGHT_DICE}·Dice + {Config.LOSS_WEIGHT_FOCAL}·Focal")
    else:
        print(f"   ├─ Modo: ❌ COMPENSACIÓN DESACTIVADA (baseline)")
        print(f"   ├─ Focal Loss α: None (sin ponderación de clases)")
        print(f"   ├─ Focal Loss γ: {Config.FOCAL_GAMMA} (down-weight ejemplos fáciles)")
        print(f"   ├─ Dice smooth: {Config.DICE_SMOOTH}")
        print(f"   └─ Pesos: 1.0·Dice + 1.0·Focal (suma simple)")

    print(f"\n🎯 Run Name: {Config.RUN_NAME}")
    print("="*80 + "\n")


# ============================================================================
# TRANSFORMACIONES (DATA AUGMENTATION)
# ============================================================================

def get_train_transforms():
    """Retorna transformaciones de entrenamiento."""
    return AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90, p=0.5, align_corners=False),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3),  # NUEVO
        K.RandomGaussianNoise(mean=0., std=0.05, p=0.3),  # NUEVO
        # Nota: ColorJitter no es compatible con imágenes multiespectrales (6 bandas)
        # Solo funciona con RGB (3 canales) o escala de grises (1 canal)
        data_keys=None,  # Kornia transforma TODOS los valores del diccionario
        same_on_batch=False,
    )


def get_val_transforms():
    """Retorna transformaciones de validación (ninguna por defecto)."""
    return None


# ============================================================================
# ANÁLISIS Y VISUALIZACIÓN DE DATOS
# ============================================================================

def setup_output_directories():
    """
    Crea la estructura de directorios para outputs.

    Crea:
        - outputs/
        - outputs/figures/
        - outputs/stats/
    """
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    os.makedirs(Config.STATS_DIR, exist_ok=True)

    print("\n" + "="*80)
    print("ESTRUCTURA DE DIRECTORIOS")
    print("="*80)
    print(f"✓ {Config.OUTPUT_DIR}/")
    print(f"  ✓ {Config.FIGURES_DIR}/")
    print(f"  ✓ {Config.STATS_DIR}/")
    print("="*80 + "\n")


def _process_dataset(dataset, dataset_name: str) -> Tuple[int, int, int]:
    """
    Función auxiliar para procesar un dataset individual y contar píxeles.

    Args:
        dataset: Dataset a procesar (train o val)
        dataset_name: Nombre del dataset para logging ("Entrenamiento" o "Validación")

    Returns:
        Tupla (mangrove_pixels, non_mangrove_pixels, total_masks)
    """
    mangrove_pixels = 0
    non_mangrove_pixels = 0
    total_masks = 0

    print(f"Procesando máscaras de {dataset_name.lower()}...")

    for idx in range(len(dataset)):
        # Obtener muestra
        sample = dataset[idx]

        # Extraer máscara según formato
        if isinstance(sample, dict):
            mask = sample['mask']
        else:
            _, mask = sample

        # Convertir a numpy si es tensor
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Remover dimensiones extra
        while mask.ndim > 2:
            mask = mask.squeeze()

        # Contar píxeles (operaciones vectorizadas)
        mangrove_pixels += np.sum(mask == 1)
        non_mangrove_pixels += np.sum(mask == 0)
        total_masks += 1

    return mangrove_pixels, non_mangrove_pixels, total_masks


def compute_pixel_statistics(dm: TorchGeoDataModule):
    """
    Calcula la distribución de píxeles por clase en train y validación.

    Genera análisis separados para:
    - Conjunto de entrenamiento
    - Conjunto de validación

    Args:
        dm: DataModule configurado con los datos

    Returns:
        Dict con estadísticas de train y val
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE DISTRIBUCIÓN DE CLASES (TRAIN + VAL)")
    print("="*80)

    # Preparar el datamodule si es necesario
    if not hasattr(dm, 'train_ds'):
        dm.setup('fit')

    # ========================================================================
    # PROCESAR CONJUNTO DE ENTRENAMIENTO
    # ========================================================================
    print("\n[1/2] CONJUNTO DE ENTRENAMIENTO")
    print("-" * 80)

    train_mangrove, train_non_mangrove, train_masks = _process_dataset(
        dm.train_ds, "Entrenamiento"
    )

    # Calcular estadísticas de entrenamiento
    train_total = train_mangrove + train_non_mangrove
    train_mangrove_pct = (train_mangrove / train_total) * 100
    train_non_mangrove_pct = (train_non_mangrove / train_total) * 100

    # Imprimir resultados de entrenamiento
    print(f"\n📊 Máscaras procesadas: {train_masks}")
    print(f"📊 Total de píxeles: {train_total:,}")
    print(f"🌿 Píxeles de manglar (clase 1): {train_mangrove:,} ({train_mangrove_pct:.2f}%)")
    print(f"🏞️  Píxeles no manglar (clase 0): {train_non_mangrove:,} ({train_non_mangrove_pct:.2f}%)")

    # Guardar estadísticas de entrenamiento
    train_stats_file = Path(Config.STATS_DIR) / 'pixel_counts_train.txt'
    with open(train_stats_file, 'w', encoding='utf-8') as f:
        f.write("DISTRIBUCIÓN DE PÍXELES POR CLASE - ENTRENAMIENTO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Máscaras procesadas: {train_masks}\n")
        f.write(f"Total de píxeles: {train_total:,}\n\n")
        f.write(f"Píxeles de manglar (clase 1): {train_mangrove:,}\n")
        f.write(f"Porcentaje de manglar: {train_mangrove_pct:.2f}%\n\n")
        f.write(f"Píxeles no manglar (clase 0): {train_non_mangrove:,}\n")
        f.write(f"Porcentaje no manglar: {train_non_mangrove_pct:.2f}%\n")

    print(f"\n✓ Estadísticas de entrenamiento guardadas en: {train_stats_file}")

    # ========================================================================
    # PROCESAR CONJUNTO DE VALIDACIÓN
    # ========================================================================
    print("\n[2/2] CONJUNTO DE VALIDACIÓN")
    print("-" * 80)

    val_mangrove, val_non_mangrove, val_masks = _process_dataset(
        dm.val_ds, "Validación"
    )

    # Calcular estadísticas de validación
    val_total = val_mangrove + val_non_mangrove
    val_mangrove_pct = (val_mangrove / val_total) * 100
    val_non_mangrove_pct = (val_non_mangrove / val_total) * 100

    # Imprimir resultados de validación
    print(f"\n📊 Máscaras procesadas: {val_masks}")
    print(f"📊 Total de píxeles: {val_total:,}")
    print(f"🌿 Píxeles de manglar (clase 1): {val_mangrove:,} ({val_mangrove_pct:.2f}%)")
    print(f"🏞️  Píxeles no manglar (clase 0): {val_non_mangrove:,} ({val_non_mangrove_pct:.2f}%)")

    # Guardar estadísticas de validación
    val_stats_file = Path(Config.STATS_DIR) / 'pixel_counts_val.txt'
    with open(val_stats_file, 'w', encoding='utf-8') as f:
        f.write("DISTRIBUCIÓN DE PÍXELES POR CLASE - VALIDACIÓN\n")
        f.write("="*80 + "\n\n")
        f.write(f"Máscaras procesadas: {val_masks}\n")
        f.write(f"Total de píxeles: {val_total:,}\n\n")
        f.write(f"Píxeles de manglar (clase 1): {val_mangrove:,}\n")
        f.write(f"Porcentaje de manglar: {val_mangrove_pct:.2f}%\n\n")
        f.write(f"Píxeles no manglar (clase 0): {val_non_mangrove:,}\n")
        f.write(f"Porcentaje no manglar: {val_non_mangrove_pct:.2f}%\n")

    print(f"\n✓ Estadísticas de validación guardadas en: {val_stats_file}")

    # ========================================================================
    # GENERAR FIGURA COMPARATIVA
    # ========================================================================
    print("\n" + "-" * 80)
    print("Generando figura comparativa...")

    generate_class_distribution_figure(
        train_mangrove=train_mangrove,
        train_non_mangrove=train_non_mangrove,
        val_mangrove=val_mangrove,
        val_non_mangrove=val_non_mangrove
    )

    # ========================================================================
    # RESUMEN COMPARATIVO
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO")
    print("="*80)
    print(f"{'':20} {'Entrenamiento':>20} {'Validación':>20}")
    print("-" * 80)
    print(f"{'Máscaras':20} {train_masks:>20,} {val_masks:>20,}")
    print(f"{'Total píxeles':20} {train_total:>20,} {val_total:>20,}")
    print(f"{'Manglar':20} {train_mangrove:>20,} {val_mangrove:>20,}")
    print(f"{'No Manglar':20} {train_non_mangrove:>20,} {val_non_mangrove:>20,}")
    print(f"{'% Manglar':20} {train_mangrove_pct:>19.2f}% {val_mangrove_pct:>19.2f}%")
    print("="*80 + "\n")

    return {
        'train': {
            'mangrove': train_mangrove,
            'non_mangrove': train_non_mangrove,
            'total': train_total,
            'masks': train_masks
        },
        'val': {
            'mangrove': val_mangrove,
            'non_mangrove': val_non_mangrove,
            'total': val_total,
            'masks': val_masks
        }
    }


def generate_class_distribution_figure(
    train_mangrove: int,
    train_non_mangrove: int,
    val_mangrove: int,
    val_non_mangrove: int
):
    """
    Genera y guarda figura comparativa de distribución de clases (Train vs Val).

    Crea una figura con 3 subplots:
    - Barras agrupadas comparando Train vs Val
    - Pie chart de entrenamiento
    - Pie chart de validación

    Args:
        train_mangrove: Píxeles de manglar en entrenamiento
        train_non_mangrove: Píxeles no-manglar en entrenamiento
        val_mangrove: Píxeles de manglar en validación
        val_non_mangrove: Píxeles no-manglar en validación
    """
    # Configurar estilo profesional
    plt.style.use('default')

    # Colores consistentes
    colors = {
        'non_mangrove': '#ff7f0e',  # Naranja
        'mangrove': '#2ca02c'        # Verde
    }

    # Crear figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # ========================================================================
    # SUBPLOT 1: BARRAS AGRUPADAS (TRAIN VS VAL)
    # ========================================================================
    x = np.arange(2)  # 2 clases
    width = 0.35

    # Datos para barras
    train_data = [train_non_mangrove, train_mangrove]
    val_data = [val_non_mangrove, val_mangrove]

    # Crear barras agrupadas
    bars1 = ax1.bar(
        x - width/2,
        train_data,
        width,
        label='Entrenamiento',
        color=['#ffb347', '#66c266'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    bars2 = ax1.bar(
        x + width/2,
        val_data,
        width,
        label='Validación',
        color=['#ff7f0e', '#2ca02c'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )

    # Configurar subplot 1
    ax1.set_ylabel('Número de Píxeles', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación Train vs Val', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['No Manglar\n(Clase 0)', 'Manglar\n(Clase 1)'], fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=10)

    # Agregar valores sobre las barras
    def autolabel(bars, ax):
        """Añade etiquetas con valores sobre las barras."""
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height):,}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    autolabel(bars1, ax1)
    autolabel(bars2, ax1)

    # ========================================================================
    # SUBPLOT 2: PIE CHART ENTRENAMIENTO
    # ========================================================================
    train_total = train_mangrove + train_non_mangrove
    train_pixels = [train_non_mangrove, train_mangrove]
    train_labels = ['No Manglar\n(Clase 0)', 'Manglar\n(Clase 1)']
    train_colors = [colors['non_mangrove'], colors['mangrove']]

    wedges2, texts2, autotexts2 = ax2.pie(
        train_pixels,
        labels=train_labels,
        colors=train_colors,
        autopct='%1.2f%%',
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        explode=(0.05, 0.05)
    )

    # Mejorar texto de porcentajes
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax2.set_title('Entrenamiento', fontsize=14, fontweight='bold', pad=15)

    # ========================================================================
    # SUBPLOT 3: PIE CHART VALIDACIÓN
    # ========================================================================
    val_total = val_mangrove + val_non_mangrove
    val_pixels = [val_non_mangrove, val_mangrove]
    val_labels = ['No Manglar\n(Clase 0)', 'Manglar\n(Clase 1)']
    val_colors = [colors['non_mangrove'], colors['mangrove']]

    wedges3, texts3, autotexts3 = ax3.pie(
        val_pixels,
        labels=val_labels,
        colors=val_colors,
        autopct='%1.2f%%',
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        explode=(0.05, 0.05)
    )

    # Mejorar texto de porcentajes
    for autotext in autotexts3:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax3.set_title('Validación', fontsize=14, fontweight='bold', pad=15)

    # ========================================================================
    # GUARDAR FIGURA
    # ========================================================================
    plt.tight_layout()

    output_path = Path(Config.FIGURES_DIR) / 'class_distribution.jpg'
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        format='jpg'
    )
    plt.close()

    print(f"✓ Figura de distribución comparativa guardada en: {output_path}")


def generate_loss_plot(
    train_losses: List[float],
    val_losses: List[float],
    best_epoch: int
):
    """
    Genera y guarda figura de convergencia del entrenamiento en alta calidad.

    Args:
        train_losses: Lista de pérdidas de entrenamiento por época
        val_losses: Lista de pérdidas de validación por época
        best_epoch: Época en la que se activó el early stopping
    """
    # Configurar estilo profesional
    plt.style.use('default')

    # Crear figura
    plt.figure(figsize=(10, 6))

    # Asegurar que ambas listas tengan el mismo tamaño
    # (puede haber diferencia por sanity check o epoch final)
    min_len = min(len(train_losses), len(val_losses))
    train_losses = train_losses[:min_len]
    val_losses = val_losses[:min_len]

    # Graficar pérdidas
    epochs = range(1, len(train_losses) + 1)
    plt.plot(
        epochs,
        train_losses,
        label='Entrenamiento',
        linewidth=2,
        color='#1f77b4',
        marker='o',
        markersize=4,
        markevery=max(1, len(epochs)//20)
    )
    plt.plot(
        epochs,
        val_losses,
        label='Validación',
        linewidth=2,
        color='#ff7f0e',
        marker='s',
        markersize=4,
        markevery=max(1, len(epochs)//20)
    )

    # Línea de early stopping
    plt.axvline(
        best_epoch + 1,  # +1 porque epochs empieza en 1
        linestyle='--',
        color='red',
        linewidth=2,
        alpha=0.7,
        label=f'Early Stop (Época {best_epoch + 1})'
    )

    # Etiquetas y título
    plt.xlabel('Época', fontsize=12, fontweight='bold')
    plt.ylabel('Pérdida (Dice + Focal)', fontsize=12, fontweight='bold')
    plt.title('Convergencia del Entrenamiento', fontsize=14, fontweight='bold', pad=15)

    # Leyenda y grid
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')

    # Mejorar ticks
    plt.tick_params(labelsize=10)

    # Ajustar límites del eje Y para mejor visualización
    # Filtrar valores NaN e Inf antes de calcular límites
    all_losses = train_losses + val_losses
    valid_losses = [l for l in all_losses if l is not None and np.isfinite(l)]
    if valid_losses:
        min_loss = min(valid_losses)
        max_loss = max(valid_losses)
        margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
        plt.ylim(min_loss - margin, max_loss + margin)
    else:
        # Si no hay valores válidos, usar límites por defecto
        print("⚠️  Advertencia: No hay valores de pérdida válidos para graficar")

    # Guardar en alta calidad
    output_path = Path(Config.FIGURES_DIR) / 'loss.jpg'
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        format='jpg'
    )
    plt.close()

    print(f"\n✓ Figura de convergencia guardada en: {output_path}")


# ============================================================================
# CALLBACK PERSONALIZADO PARA REGISTRO DE PÉRDIDAS
# ============================================================================

class LossHistoryCallback(Callback):
    """
    Callback personalizado para registrar pérdidas durante el entrenamiento.

    Captura:
        - Pérdidas de entrenamiento por época
        - Pérdidas de validación por época
        - Época del mejor modelo (early stopping)
    """

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.best_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        """Captura la pérdida de entrenamiento al final de cada época."""
        # Obtener pérdida promedio de la época
        train_loss = trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.train_losses.append(float(train_loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Captura la pérdida de validación al final de cada época."""
        # Obtener pérdida de validación
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(float(val_loss))

    def on_train_end(self, trainer, pl_module):
        """
        Al finalizar el entrenamiento, genera la figura de convergencia.
        """
        # Determinar la mejor época
        # Si hay early stopping, usar su época; sino, la última
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                # La época actual menos la paciencia es aproximadamente la mejor
                self.best_epoch = max(0, trainer.current_epoch - callback.wait_count)
                break
        else:
            # No hay early stopping, usar la última época
            self.best_epoch = trainer.current_epoch

        # Generar figura si hay datos
        if self.train_losses and self.val_losses:
            print("\n" + "="*80)
            print("GENERANDO FIGURA DE CONVERGENCIA")
            print("="*80)
            generate_loss_plot(self.train_losses, self.val_losses, self.best_epoch)
            print("="*80 + "\n")


# ============================================================================
# RANDOM FOREST - EXTRACCIÓN DE FEATURES Y CLASIFICACIÓN
# ============================================================================

class PixelFeatureExtractor:
    """
    Extractor de features por píxel para Random Forest.

    Extrae tres tipos de features:
    1. Bandas espectrales originales (6 bandas de Sentinel-2)
    2. Índices espectrales derivados (NDVI, NDWI, EVI, SAVI, MNDWI, NIR/Red)
    3. Contexto espacial mediante ventana deslizante (NxN píxeles)

    DIFERENCIAS METODOLÓGICAS CON CNNs:
    - CNNs: Aprenden features automáticamente mediante convoluciones jerárquicas
    - RF: Requiere features diseñadas manualmente por experto en teledetección
    - CNNs: Contexto espacial mediante receptive field creciente
    - RF: Contexto espacial limitado a ventana local fija

    Args:
        window_size: Tamaño de ventana para contexto espacial (ej: 5 = ventana 5x5)
        use_indices: Si True, calcula índices espectrales adicionales
    """

    def __init__(self, window_size=5, use_indices=True):
        self.window_size = window_size
        self.use_indices = use_indices
        self.pad_size = window_size // 2

    def compute_spectral_indices(self, image):
        """
        Calcula índices espectrales para imágenes Sentinel-2.

        Bandas de Sentinel-2:
        - image[:, :, 0]: B2 (Blue)
        - image[:, :, 1]: B3 (Green)
        - image[:, :, 2]: B4 (Red)
        - image[:, :, 3]: B8 (NIR)
        - image[:, :, 4]: B11 (SWIR1)
        - image[:, :, 5]: B12 (SWIR2)

        Índices calculados:
        1. NDVI: (NIR - Red) / (NIR + Red) - Vegetación vigorosa
        2. NDWI: (Green - NIR) / (Green + NIR) - Contenido de agua
        3. EVI: Enhanced Vegetation Index - Vegetación con corrección atmosférica
        4. SAVI: Soil Adjusted Vegetation Index - Vegetación ajustada por suelo
        5. MNDWI: Modified NDWI - Agua y humedad
        6. NIR/Red Ratio: Ratio simple NIR/Red

        Args:
            image: (H, W, 6) - imagen Sentinel-2

        Returns:
            indices_array: (H, W, 6) - índices espectrales
        """
        b2 = image[:, :, 0]   # Blue
        b3 = image[:, :, 1]   # Green
        b4 = image[:, :, 2]   # Red
        b8 = image[:, :, 3]   # NIR
        b11 = image[:, :, 4]  # SWIR1
        b12 = image[:, :, 5]  # SWIR2

        eps = 1e-8  # Evitar divisiones por cero

        indices = []

        # 1. NDVI (Normalized Difference Vegetation Index)
        ndvi = (b8 - b4) / (b8 + b4 + eps)
        indices.append(ndvi)

        # 2. NDWI (Normalized Difference Water Index)
        ndwi = (b3 - b8) / (b3 + b8 + eps)
        indices.append(ndwi)

        # 3. EVI (Enhanced Vegetation Index)
        evi = 2.5 * ((b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1 + eps))
        indices.append(evi)

        # 4. SAVI (Soil Adjusted Vegetation Index)
        L = 0.5  # Factor de ajuste de suelo
        savi = ((b8 - b4) / (b8 + b4 + L + eps)) * (1 + L)
        indices.append(savi)

        # 5. MNDWI (Modified Normalized Difference Water Index)
        # Especialmente útil para manglares (ecosistemas agua-tierra)
        mndwi = (b3 - b11) / (b3 + b11 + eps)
        indices.append(mndwi)

        # 6. NIR/Red Ratio
        nir_red_ratio = b8 / (b4 + eps)
        indices.append(nir_red_ratio)

        # Stack todos los índices
        indices_array = np.stack(indices, axis=-1)  # (H, W, 6)
        return indices_array

    def extract_spatial_context(self, image):
        """
        Extrae contexto espacial usando ventana deslizante.

        LIMITACIÓN: A diferencia de CNNs que tienen receptive fields grandes y
        aprenden automáticamente qué contexto es relevante, RF usa ventanas fijas
        y pequeñas. Esto limita su capacidad de capturar patrones espaciales complejos.

        Args:
            image: (H, W, C) - imagen multiespectral o índices

        Returns:
            features: (H, W, C * window_size^2) - features con contexto espacial
        """
        if self.window_size <= 1:
            return image

        # Validación de dimensiones
        if image.ndim != 3:
            raise ValueError(
                f"extract_spatial_context espera imagen 3D (H, W, C), "
                f"pero recibió shape: {image.shape} con ndim={image.ndim}"
            )

        H, W, C = image.shape

        # Pad con reflexión para bordes
        padded = np.pad(
            image,
            ((self.pad_size, self.pad_size),
             (self.pad_size, self.pad_size),
             (0, 0)),
            mode='reflect'
        )

        # Extraer ventanas para cada píxel
        features_list = []

        for i in range(H):
            for j in range(W):
                # Ventana centrada en (i, j)
                window = padded[
                    i:i+self.window_size,
                    j:j+self.window_size,
                    :
                ]
                # Aplanar ventana a vector
                window_flat = window.flatten()
                features_list.append(window_flat)

        # Reshape a (H, W, total_features)
        features = np.array(features_list).reshape(H, W, -1)
        return features

    def extract_features(self, image):
        """
        Pipeline completo de extracción de features.

        Combina:
        1. Bandas originales (6 features)
        2. Índices espectrales (6 features) - opcional
        3. Contexto espacial de bandas (6 * window_size^2 features)
        4. Contexto espacial de índices (6 * window_size^2 features) - opcional

        Total de features (con window_size=5, use_indices=True):
        - Bandas: 6
        - Índices: 6
        - Contexto bandas: 6 * 25 = 150
        - Contexto índices: 6 * 25 = 150
        - TOTAL: 312 features

        Args:
            image: (H, W, 6) - imagen Sentinel-2

        Returns:
            features: (H, W, num_features) - todas las features concatenadas
        """
        # === VALIDACIÓN DE ENTRADA ===
        if not isinstance(image, np.ndarray):
            raise TypeError(f"extract_features espera numpy array, recibió: {type(image)}")

        if image.ndim != 3:
            raise ValueError(
                f"extract_features espera imagen 3D (H, W, C), "
                f"pero recibió shape: {image.shape} con ndim={image.ndim}"
            )

        if image.shape[-1] not in [6, Config.TOTAL_CHANNELS]:
            raise ValueError(
                f"extract_features espera {Config.TOTAL_CHANNELS} canales, "
                f"pero recibió shape: {image.shape}"
            )

        features_list = []

        # 1. Bandas originales (siempre)
        features_list.append(image)

        # 2. Índices espectrales
        if self.use_indices:
            indices = self.compute_spectral_indices(image)
            features_list.append(indices)

        # 3. Contexto espacial
        if self.window_size > 1:
            # Contexto de bandas originales
            spatial_context = self.extract_spatial_context(image)
            features_list.append(spatial_context)

            # Contexto de índices espectrales
            if self.use_indices:
                indices_context = self.extract_spatial_context(indices)
                features_list.append(indices_context)

        # Concatenar todas las features
        features = np.concatenate(features_list, axis=-1)
        return features


class RandomForestSegmentationModule:
    """
    Wrapper de Random Forest para segmentación semántica.

    Random Forest es un ensemble de árboles de decisión que clasifica cada píxel
    independientemente basándose en sus features espectrales y de contexto.

    VENTAJAS vs CNNs:
    - No requiere GPU (CPU-only)
    - Entrenamiento más rápido (minutos vs horas)
    - Interpretable (feature importance)
    - Robusto con pocos datos
    - No requiere data augmentation

    LIMITACIONES vs CNNs:
    - Sin aprendizaje de features (depende de ingeniería manual)
    - Contexto espacial limitado (ventana pequeña vs receptive field grande)
    - Clasificación píxel a píxel (no holística)
    - Predicción más lenta por píxel en inferencia
    - No captura patrones espaciales complejos

    Args:
        n_estimators: Número de árboles en el bosque
        max_depth: Profundidad máxima de cada árbol
        min_samples_split: Mínimo de muestras para dividir nodo
        min_samples_leaf: Mínimo de muestras en hoja
        window_size: Tamaño de ventana de contexto
        use_indices: Calcular índices espectrales
        n_jobs: Número de cores para paralelización
        random_state: Semilla para reproducibilidad
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        window_size=5,
        use_indices=True,
        n_jobs=-1,
        random_state=42
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1,
            class_weight='balanced',  # Compensa desbalance de clases
            bootstrap=True,
            oob_score=True  # Out-of-bag score para validación
        )

        self.feature_extractor = PixelFeatureExtractor(
            window_size=window_size,
            use_indices=use_indices
        )

        self.is_fitted = False
        self.n_features = None

    def prepare_data(self, images, masks):
        """
        Prepara datos para entrenamiento de Random Forest.

        Convierte imágenes y máscaras a formato (n_samples, n_features):
        - Cada píxel se convierte en una muestra
        - Cada feature se convierte en una columna

        Args:
            images: Lista de arrays (H, W, 6)
            masks: Lista de arrays (H, W)

        Returns:
            X: (N_pixels, N_features) - matriz de features
            y: (N_pixels,) - vector de labels
        """
        X_list = []
        y_list = []

        print(f"Preparando datos: {len(images)} imágenes...")

        for idx, (image, mask) in enumerate(zip(images, masks)):
            if idx % 50 == 0:
                print(f"  Procesando imagen {idx}/{len(images)}...")

            # Extraer features
            features = self.feature_extractor.extract_features(image)

            # Aplanar a formato (n_pixels, n_features)
            H, W, F = features.shape
            X_flat = features.reshape(-1, F)
            y_flat = mask.reshape(-1)

            X_list.append(X_flat)
            y_list.append(y_flat)

        # Concatenar todas las imágenes
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        self.n_features = X.shape[1]

        print(f"\nDatos preparados:")
        print(f"  Shape X: {X.shape}")
        print(f"  Shape y: {y.shape}")
        print(f"  Features por píxel: {self.n_features}")

        # Estadísticas de clases
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Distribución de clases:")
        for cls, count in zip(unique, counts):
            pct = (count / len(y)) * 100
            print(f"    Clase {int(cls)}: {count:,} ({pct:.2f}%)")

        return X, y

    def fit(self, train_images, train_masks, val_images=None, val_masks=None, max_samples=None):
        """
        Entrena el modelo Random Forest.

        Args:
            train_images: Lista de imágenes de entrenamiento
            train_masks: Lista de máscaras de entrenamiento
            val_images: Lista de imágenes de validación (opcional)
            val_masks: Lista de máscaras de validación (opcional)
            max_samples: Máximo de píxeles para entrenamiento (submuestreo)
        """
        print("\n" + "="*80)
        print("ENTRENAMIENTO RANDOM FOREST")
        print("="*80)

        # Preparar datos de entrenamiento
        X_train, y_train = self.prepare_data(train_images, train_masks)

        # Submuestreo si hay demasiados píxeles
        if max_samples and len(X_train) > max_samples:
            print(f"\n⚠️  Submuestreando de {len(X_train):,} a {max_samples:,} píxeles...")
            print(f"    (Para acelerar entrenamiento. Desactivar submuestreo para mejor rendimiento)")

            # Submuestreo estratificado para mantener balance de clases
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train,
                train_size=max_samples,
                stratify=y_train,
                random_state=42
            )

            print(f"    Nuevo shape X: {X_train.shape}")
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"    Nueva distribución:")
            for cls, count in zip(unique, counts):
                pct = (count / len(y_train)) * 100
                print(f"      Clase {int(cls)}: {count:,} ({pct:.2f}%)")

        # Entrenar Random Forest
        print(f"\n🌲 Entrenando Random Forest con {self.rf.n_estimators} árboles...")
        print(f"   (Esto puede tardar varios minutos dependiendo del dataset)")

        self.rf.fit(X_train, y_train)
        self.is_fitted = True

        print("✅ Entrenamiento completado!")

        # Out-of-bag score
        if self.rf.oob_score:
            print(f"\n📊 Out-of-Bag Score: {self.rf.oob_score_:.4f}")

        # Feature importance (top 10)
        print(f"\n📈 Top 10 Features más importantes:")
        importances = self.rf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for rank, idx in enumerate(indices, 1):
            print(f"   {rank}. Feature {idx}: {importances[idx]:.4f}")

        # Métricas en train
        print(f"\n🎯 Evaluando en training set...")
        y_pred_train = self.rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"   Train Accuracy: {train_acc:.4f}")

        # Matriz de confusión en train
        cm_train = confusion_matrix(y_train, y_pred_train)
        print(f"   Confusion Matrix (Train):")
        print(f"   {cm_train}")

        # Validación si se proporciona
        if val_images is not None and val_masks is not None:
            print(f"\n🎯 Evaluando en validation set...")
            X_val, y_val = self.prepare_data(val_images, val_masks)
            y_pred_val = self.rf.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            print(f"   Val Accuracy: {val_acc:.4f}")

            # Matriz de confusión en val
            cm_val = confusion_matrix(y_val, y_pred_val)
            print(f"   Confusion Matrix (Val):")
            print(f"   {cm_val}")

        print("="*80 + "\n")

        return self

    def predict(self, image):
        """
        Predice máscara de segmentación para una imagen.

        Args:
            image: (H, W, 6) - imagen Sentinel-2

        Returns:
            mask: (H, W) - máscara predicha (0 o 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")

        H, W, _ = image.shape

        # Extraer features
        features = self.feature_extractor.extract_features(image)

        # Aplanar
        X = features.reshape(-1, features.shape[-1])

        # Predecir
        y_pred = self.rf.predict(X)

        # Reshape a imagen
        mask = y_pred.reshape(H, W)

        return mask.astype(np.float32)

    def predict_batch(self, images):
        """Predice máscaras para un lote de imágenes."""
        masks = []
        for idx, image in enumerate(images):
            if idx % 50 == 0:
                print(f"  Prediciendo imagen {idx}/{len(images)}...")
            mask = self.predict(image)
            masks.append(mask)
        return np.array(masks)

    def save(self, filepath):
        """Guarda el modelo entrenado."""
        save_dict = {
            'rf': self.rf,
            'feature_extractor': self.feature_extractor,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features
        }
        joblib.dump(save_dict, filepath)
        print(f"💾 Modelo Random Forest guardado en: {filepath}")

    def load(self, filepath):
        """Carga un modelo entrenado."""
        save_dict = joblib.load(filepath)
        self.rf = save_dict['rf']
        self.feature_extractor = save_dict['feature_extractor']
        self.is_fitted = save_dict['is_fitted']
        self.n_features = save_dict['n_features']
        print(f"📂 Modelo Random Forest cargado de: {filepath}")


def extract_dataset_as_numpy(dataset):
    """
    Convierte TorchGeo dataset a listas de numpy arrays.

    Args:
        dataset: TorchGeo dataset

    Returns:
        images: Lista de arrays (H, W, 6)
        masks: Lista de arrays (H, W)
    """
    images = []
    masks = []

    for idx in range(len(dataset)):
        sample = dataset[idx]

        # Extraer imagen y máscara del diccionario
        if isinstance(sample, dict):
            image = sample['image']
            mask = sample['mask']
        else:
            image, mask = sample

        # Convertir tensors a numpy
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # === MANEJO ROBUSTO DE DIMENSIONES DE IMAGEN ===

        # Eliminar dimensiones de tamaño 1 (batch dimension si existe)
        image = np.squeeze(image)

        # Verificar y ajustar dimensiones
        if image.ndim == 2:
            # Caso: imagen en escala de grises (H, W) -> agregar canal
            image = np.expand_dims(image, axis=-1)
        elif image.ndim == 3:
            # Caso normal: (C, H, W) o (H, W, C)
            # Detectar si necesita transposición
            if image.shape[0] in [6, Config.TOTAL_CHANNELS]:
                # Formato (C, H, W) -> transponer a (H, W, C)
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[-1] in [6, Config.TOTAL_CHANNELS]:
                # Ya está en formato (H, W, C) - no hacer nada
                pass
            else:
                # Último recurso: asumir que primera dim pequeña es canales
                if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                    image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 4:
            # Caso: imagen con batch dimension (B, C, H, W) o (B, H, W, C)
            # Eliminar batch dimension (debería ser 1)
            if image.shape[0] == 1:
                image = image[0]  # (C, H, W) o (H, W, C)
                # Verificar si necesita transposición
                if image.shape[0] in [6, Config.TOTAL_CHANNELS]:
                    image = np.transpose(image, (1, 2, 0))
            else:
                raise ValueError(f"Imagen con batch size != 1: {image.shape}")
        else:
            raise ValueError(f"Dimensiones inesperadas para imagen: {image.shape}")

        # Verificación final: debe ser (H, W, C)
        if image.ndim != 3:
            raise ValueError(f"Error: imagen no tiene 3 dimensiones después de procesamiento. Shape: {image.shape}")

        if image.shape[-1] not in [6, Config.TOTAL_CHANNELS]:
            raise ValueError(f"Error: número de canales incorrecto. Shape: {image.shape}, esperado: (H, W, {Config.TOTAL_CHANNELS})")

        # === MANEJO DE MÁSCARA ===

        # Eliminar dimensiones extra de máscara
        mask = np.squeeze(mask)

        # Asegurar que es 2D
        if mask.ndim > 2:
            # Si aún tiene más dimensiones, tomar el primer canal
            mask = mask[0] if mask.shape[0] == 1 else mask
            mask = np.squeeze(mask)

        if mask.ndim != 2:
            raise ValueError(f"Error: máscara no es 2D después de procesamiento. Shape: {mask.shape}")

        images.append(image.astype(np.float32))
        masks.append(mask.astype(np.int32))

    return images, masks


def compute_rf_segmentation_metrics(predictions, targets):
    """
    Calcula métricas de segmentación para Random Forest.

    Usa las mismas métricas que los modelos CNN para comparación justa:
    - IoU (Intersection over Union)
    - F1-Score
    - Kappa (Cohen's Kappa)
    - Overall Accuracy
    - Precision
    - Recall
    - FWIoU (Frequency Weighted IoU)

    Args:
        predictions: Lista de máscaras predichas (H, W)
        targets: Lista de máscaras ground truth (H, W)

    Returns:
        Dict con todas las métricas
    """
    # Convertir a tensors para usar funciones existentes del módulo
    preds_list = []
    targets_list = []

    for pred, target in zip(predictions, targets):
        # Agregar dimensión de canal: (H, W) -> (1, 1, H, W)
        pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        target_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)

        preds_list.append(pred_tensor)
        targets_list.append(target_tensor)

    # Concatenar batch
    preds_batch = torch.cat(preds_list, dim=0)
    targets_batch = torch.cat(targets_list, dim=0)

    # Calcular métricas usando funciones del módulo
    metrics_dict = {
        'iou': float(iou(preds_batch, targets_batch)),
        'f1': float(f1_score(preds_batch, targets_batch)),
        'kappa': float(kappa(preds_batch, targets_batch)),
        'oa': float(overall_accuracy(preds_batch, targets_batch)),
        'precision': float(precision(preds_batch, targets_batch)),
        'recall': float(recall(preds_batch, targets_batch)),
        'fwiou': float(fw_iou(preds_batch, targets_batch)),
    }

    return metrics_dict


def train_random_forest_pipeline(dm: TorchGeoDataModule):
    """
    Pipeline completo de entrenamiento y evaluación de Random Forest.

    Este pipeline es independiente de PyTorch Lightning y maneja:
    1. Creación del modelo RF
    2. Extracción de datos del DataModule
    3. Entrenamiento
    4. Evaluación en train y val
    5. Cálculo de métricas
    6. Guardado de modelo y resultados

    Args:
        dm: TorchGeo DataModule configurado

    Returns:
        model: Modelo Random Forest entrenado
        metrics: Dict con métricas de validación
    """
    print("\n" + "="*80)
    print("PIPELINE RANDOM FOREST - BASELINE CLÁSICO DE MACHINE LEARNING")
    print("="*80)
    print("\n📋 DIFERENCIAS METODOLÓGICAS:")
    print("   CNNs: Aprenden features automáticamente mediante convoluciones")
    print("   RF:   Usa features diseñadas manualmente (bandas + índices + contexto)")
    print("\n📋 LIMITACIONES:")
    print("   - Clasificación píxel a píxel (sin visión holística)")
    print("   - Contexto espacial limitado a ventana local")
    print("   - No captura patrones espaciales complejos")
    print("\n📋 VENTAJAS:")
    print("   - No requiere GPU")
    print("   - Entrenamiento rápido")
    print("   - Interpretable (feature importance)")
    print("="*80 + "\n")

    # Preparar datasets
    dm.setup('fit')

    # Extraer datos como numpy arrays
    print("📦 Extrayendo imágenes de entrenamiento...")
    train_images, train_masks = extract_dataset_as_numpy(dm.train_ds)
    print(f"   Train: {len(train_images)} imágenes")

    print("\n📦 Extrayendo imágenes de validación...")
    val_images, val_masks = extract_dataset_as_numpy(dm.val_ds)
    print(f"   Val: {len(val_images)} imágenes")

    # Crear modelo
    print("\n🌲 Creando Random Forest Classifier...")
    model = RandomForestSegmentationModule(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        window_size=Config.RF_WINDOW_SIZE,
        use_indices=Config.RF_USE_INDICES,
        n_jobs=Config.RF_N_JOBS,
        random_state=Config.RF_RANDOM_STATE
    )

    # Entrenar
    model.fit(
        train_images,
        train_masks,
        val_images,
        val_masks,
        max_samples=Config.RF_MAX_SAMPLES
    )

    # Evaluar en validación
    print("\n" + "="*80)
    print("EVALUACIÓN EN CONJUNTO DE VALIDACIÓN")
    print("="*80)

    print("\n🔮 Prediciendo máscaras de validación...")
    val_preds = model.predict_batch(val_images)

    print("\n📊 Calculando métricas...")
    metrics_dict = compute_rf_segmentation_metrics(val_preds, val_masks)

    print("\n" + "="*80)
    print("MÉTRICAS DE VALIDACIÓN - RANDOM FOREST")
    print("="*80)
    print(f"  IoU (Intersection over Union): {metrics_dict['iou']:.4f}")
    print(f"  F1-Score:                      {metrics_dict['f1']:.4f}")
    print(f"  Kappa (Cohen's):               {metrics_dict['kappa']:.4f}")
    print(f"  Overall Accuracy:              {metrics_dict['oa']:.4f}")
    print(f"  Precision:                     {metrics_dict['precision']:.4f}")
    print(f"  Recall:                        {metrics_dict['recall']:.4f}")
    print(f"  FWIoU:                         {metrics_dict['fwiou']:.4f}")
    print("="*80 + "\n")

    # Guardar modelo
    model_path = Path(Config.CHECKPOINT_DIR) / 'random_forest_model.pkl'
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    model.save(model_path)

    # Guardar métricas en archivo
    metrics_file = Path(Config.STATS_DIR) / 'random_forest_metrics.txt'
    os.makedirs(Config.STATS_DIR, exist_ok=True)

    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("MÉTRICAS DE VALIDACIÓN - RANDOM FOREST\n")
        f.write("="*80 + "\n\n")
        f.write(f"Modelo: Random Forest Classifier\n")
        f.write(f"N° Árboles: {Config.RF_N_ESTIMATORS}\n")
        f.write(f"Max Depth: {Config.RF_MAX_DEPTH}\n")
        f.write(f"Window Size: {Config.RF_WINDOW_SIZE}x{Config.RF_WINDOW_SIZE}\n")
        f.write(f"Índices Espectrales: {'Sí' if Config.RF_USE_INDICES else 'No'}\n")
        f.write(f"Features por píxel: {model.n_features}\n\n")
        f.write("MÉTRICAS:\n")
        f.write(f"  IoU:       {metrics_dict['iou']:.4f}\n")
        f.write(f"  F1-Score:  {metrics_dict['f1']:.4f}\n")
        f.write(f"  Kappa:     {metrics_dict['kappa']:.4f}\n")
        f.write(f"  Accuracy:  {metrics_dict['oa']:.4f}\n")
        f.write(f"  Precision: {metrics_dict['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics_dict['recall']:.4f}\n")
        f.write(f"  FWIoU:     {metrics_dict['fwiou']:.4f}\n")

    print(f"💾 Métricas guardadas en: {metrics_file}")

    # Guardar métricas en JSON para comparación programática
    json_file = Path(Config.STATS_DIR) / 'random_forest_metrics.json'
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"💾 Métricas JSON guardadas en: {json_file}")

    return model, metrics_dict


# ============================================================================
# DATAMODULE
# ============================================================================

def create_datamodule():
    """Crea y retorna el DataModule configurado."""
    dm = TorchGeoDataModule(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        train_trans=get_train_transforms(),
        val_trans=get_val_transforms(),
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        bands=Config.BANDS,
        norm_value=Config.NORM_VALUE,
        cache_data=Config.CACHE_DATA,
        verbose=True,
    )
    
    # Validar número de bandas
    assert dm.num_bands == Config.TOTAL_CHANNELS, \
        f"Mismatch: dm.num_bands={dm.num_bands}, expected={Config.TOTAL_CHANNELS}"
    
    return dm


# ============================================================================
# MODELO Y OPTIMIZACIÓN
# ============================================================================

def create_model():
    """
    Crea y retorna el modelo según Config.MODEL_TYPE.

    Modelos disponibles:
    - 'multi_branch': MultiBranchUNetWrapper (UNet++ con fusión multi-resolución)
    - 'unet': UNet básico estándar
    - 'unet_inception': UNet con encoder InceptionResNetV2
    - 'deeplabv3plus': DeepLabV3+ estándar de SMP

    Returns:
        Modelo de segmentación configurado
    """
    if Config.MODEL_TYPE == "deeplabv3plus":
        # ================================================================
        # DeepLabV3+ estándar (SMP)
        # ================================================================
        model = smp.DeepLabV3Plus(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.TOTAL_CHANNELS,
            classes=Config.NUM_CLASSES,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            upsampling=4,
            activation=None,
        )
        print("\n" + "=" * 70)
        print("DeepLabV3+ Inicializado")
        print("=" * 70)
        print(f"Encoder: {Config.ENCODER}")
        print(f"In Channels: {Config.TOTAL_CHANNELS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Output Stride: 16")
        print(f"ASPP Rates: (12, 24, 36)")
        print("=" * 70 + "\n")

    elif Config.MODEL_TYPE == "unet":
        # ================================================================
        # UNet básico (SMP)
        # ================================================================
        model = smp.Unet(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.TOTAL_CHANNELS,
            classes=Config.NUM_CLASSES,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            activation=None,
        )
        print("\n" + "=" * 70)
        print("UNet Básico Inicializado")
        print("=" * 70)
        print(f"Encoder: {Config.ENCODER}")
        print(f"In Channels: {Config.TOTAL_CHANNELS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Decoder Channels: (256, 128, 64, 32, 16)")
        print("=" * 70 + "\n")

    elif Config.MODEL_TYPE == "unet_inception":
        # ================================================================
        # UNet con InceptionResNetV2 (Google Inception mejorado)
        # ================================================================
        model = smp.Unet(
            encoder_name="inceptionresnetv2",  # Encoder fijo
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.TOTAL_CHANNELS,
            classes=Config.NUM_CLASSES,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            activation=None,
        )
        print("\n" + "=" * 70)
        print("UNet + InceptionResNetV2 Inicializado")
        print("=" * 70)
        print(f"Encoder: inceptionresnetv2 (Google Inception mejorado)")
        print(f"In Channels: {Config.TOTAL_CHANNELS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Decoder Channels: (256, 128, 64, 32, 16)")
        print(f"Nota: InceptionResNetV2 combina Inception modules con conexiones residuales")
        print("=" * 70 + "\n")

    elif Config.MODEL_TYPE == "unetplusplus":
        # ================================================================
        # UNet++ estándar (SMP) - Nested Skip Connections
        # ================================================================
        model = smp.UnetPlusPlus(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.TOTAL_CHANNELS,
            classes=Config.NUM_CLASSES,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            activation=None,
        )
        print("\n" + "=" * 70)
        print("UNet++ (Nested U-Net) Inicializado")
        print("=" * 70)
        print(f"Encoder: {Config.ENCODER}")
        print(f"In Channels: {Config.TOTAL_CHANNELS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Decoder Channels: (256, 128, 64, 32, 16)")
        print(f"Nota: UNet++ usa nested skip connections para mejor propagación de features")
        print("=" * 70 + "\n")

    elif Config.MODEL_TYPE == "hrnet":
        # ================================================================
        # HRNet (High-Resolution Net) con UNet decoder
        # ================================================================
        # HRNet en SMP se puede usar con diferentes decoders
        # Usamos UNet decoder para consistencia con otros modelos
        model = smp.Unet(
            encoder_name=Config.ENCODER,  # hrnet_w18, hrnet_w32, hrnet_w40, hrnet_w48
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.TOTAL_CHANNELS,
            classes=Config.NUM_CLASSES,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
            activation=None,
        )
        print("\n" + "=" * 70)
        print("HRNet (High-Resolution Net) Inicializado")
        print("=" * 70)
        print(f"Encoder: {Config.ENCODER}")
        print(f"In Channels: {Config.TOTAL_CHANNELS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Decoder: UNet (Skip Connections)")
        print(f"Decoder Channels: (256, 128, 64, 32, 16)")
        print(f"Nota: HRNet mantiene representaciones de alta resolución en paralelo")
        print(f"      durante todo el proceso, ideal para segmentación fina")
        print("=" * 70 + "\n")

    elif Config.MODEL_TYPE == "pspnet":
        # ================================================================
        # PSPNet (Pyramid Scene Parsing Network)
        # ================================================================
        model = smp.PSPNet(
            encoder_name=Config.ENCODER,  # resnet50, resnet101 recomendados
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.TOTAL_CHANNELS,
            classes=Config.NUM_CLASSES,
            psp_out_channels=512,  # Canales de salida del módulo PSP
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            activation=None,
        )
        print("\n" + "=" * 70)
        print("PSPNet (Pyramid Scene Parsing Network) Inicializado")
        print("=" * 70)
        print(f"Encoder: {Config.ENCODER}")
        print(f"In Channels: {Config.TOTAL_CHANNELS}")
        print(f"Classes: {Config.NUM_CLASSES}")
        print(f"Decoder: Pyramid Pooling Module (PPM)")
        print(f"PSP Output Channels: 512")
        print(f"PSP Pooling Sizes: (1, 2, 3, 6)")
        print(f"PSP Dropout: 0.2")
        print(f"Nota: PSPNet captura contexto multi-escala mediante pyramid pooling")
        print(f"      Especialmente efectivo con ResNet101 para escenas complejas")
        print("=" * 70 + "\n")

    elif Config.MODEL_TYPE == "segformer":
        # ================================================================
        # SegFormer (Vision Transformer)
        # ================================================================
        model = create_segformer_model(
            variant=Config.SEGFORMER_VARIANT,
            in_channels=Config.TOTAL_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            pretrained=Config.SEGFORMER_PRETRAINED,
            use_aux_loss=Config.SEGFORMER_USE_AUX_LOSS,
            aux_weight=Config.SEGFORMER_AUX_WEIGHT
        )
        # Nota: La información del modelo ya se imprime dentro de create_segformer_model()

    elif Config.MODEL_TYPE == "segnet":
        # ================================================================
        # SegNet (Arquitectura clásica encoder-decoder)
        # ================================================================
        model = create_segnet_model(
            variant=Config.SEGNET_VARIANT,
            in_channels=Config.TOTAL_CHANNELS,
            num_classes=Config.NUM_CLASSES
        )
        # Nota: La información del modelo ya se imprime dentro de create_segnet_model()

    elif Config.MODEL_TYPE == "segnext":
        # ================================================================
        # SegNeXt (CNN moderna SOTA con atención multi-escala)
        # ================================================================
        model = create_segnext_model(
            variant=Config.SEGNEXT_VARIANT,
            in_channels=Config.TOTAL_CHANNELS,
            num_classes=Config.NUM_CLASSES
        )
        # Nota: La información del modelo ya se imprime dentro de create_segnext_model()

    elif Config.MODEL_TYPE == "mask2former":
        # ================================================================
        # Mask2Former (Transformer SOTA con masked attention)
        # ================================================================
        model = create_mask2former_model(
            variant=Config.MASK2FORMER_VARIANT,
            in_channels=Config.TOTAL_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            pretrained=True
        )
        # Nota: La información del modelo ya se imprime dentro de create_mask2former_model()

    elif Config.MODEL_TYPE == "swinv2":
        # ================================================================
        # Swin Transformer V2 + UPerNet/UNet (Transformer jerárquico SOTA)
        # ================================================================
        model = create_swinv2_model(
            variant=Config.SWINV2_VARIANT,
            decoder=Config.SWINV2_DECODER,
            in_channels=Config.TOTAL_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            image_size=Config.TILE_SIZE
        )
        # Nota: La información del modelo ya se imprime dentro de create_swinv2_model()

    elif Config.MODEL_TYPE == "random_forest":
        # ================================================================
        # Random Forest Classifier (Baseline clásico de ML)
        # ================================================================
        # Nota: Random Forest no se retorna aquí ya que requiere pipeline especial
        # El entrenamiento se maneja en train_random_forest_pipeline()
        # Este caso existe para validación de configuración
        print("\n" + "=" * 70)
        print("Random Forest Classifier Configurado")
        print("=" * 70)
        print(f"⚠️  Nota: Random Forest usa pipeline especial (no PyTorch)")
        print(f"N° Árboles: {Config.RF_N_ESTIMATORS}")
        print(f"Max Depth: {Config.RF_MAX_DEPTH}")
        print(f"Window Size: {Config.RF_WINDOW_SIZE}x{Config.RF_WINDOW_SIZE}")
        print(f"Use Indices: {Config.RF_USE_INDICES}")

        # Calcular número de features
        n_bands = 6
        n_indices = 6 if Config.RF_USE_INDICES else 0
        n_context = (Config.RF_WINDOW_SIZE ** 2) * n_bands if Config.RF_WINDOW_SIZE > 1 else 0
        n_context_idx = (Config.RF_WINDOW_SIZE ** 2) * n_indices if (Config.RF_WINDOW_SIZE > 1 and Config.RF_USE_INDICES) else 0
        total_features = n_bands + n_indices + n_context + n_context_idx

        print(f"\nFeatures por píxel: {total_features}")
        print(f"  - Bandas originales: {n_bands}")
        if n_indices > 0:
            print(f"  - Índices espectrales: {n_indices}")
        if n_context > 0:
            print(f"  - Contexto espacial (bandas): {n_context}")
        if n_context_idx > 0:
            print(f"  - Contexto espacial (índices): {n_context_idx}")
        print("=" * 70 + "\n")

        # Retornar None - el modelo se crea en el pipeline especial
        return None

    else:
        # ================================================================
        # Multi-Branch UNet++ (default)
        # ================================================================
        model = MultiBranchUNetWrapper(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            high_res_channels=Config.HIGH_RES_CHANNELS,
            low_res_channels=Config.LOW_RES_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            fusion_mode=Config.FUSION_MODE,
            upsample_mode=Config.UPSAMPLE_MODE,
            deep_supervision=Config.DEEP_SUPERVISION,
        )

    return model


def create_optimizer(model):
    """Crea y retorna el optimizador."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
    )
    return optimizer


def create_scheduler(optimizer):
    """
    Crea y retorna el scheduler.

    Para Transformers que entrenan desde cero (como Swin V2), se usa warmup lineal
    seguido de ReduceLROnPlateau.
    """
    if Config.WARMUP_EPOCHS > 0:
        # Usar warmup: empezar con LR muy bajo y subir gradualmente
        print(f"⚡ Warmup activado: {Config.WARMUP_EPOCHS} épocas")

        # Scheduler con warmup lineal
        def warmup_lambda(epoch):
            if epoch < Config.WARMUP_EPOCHS:
                # Warmup lineal: de 0.1*LR a LR
                return 0.1 + 0.9 * (epoch / Config.WARMUP_EPOCHS)
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda
        )
        return warmup_scheduler
    else:
        # Sin warmup: usar ReduceLROnPlateau directamente
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        return scheduler


def create_loss_fn():
    """
    Crea y retorna la función de pérdida.

    El comportamiento depende del flag Config.USE_CLASS_BALANCING:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ USE_CLASS_BALANCING = True (Compensación EXPLÍCITA)                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ - Focal Loss: alpha=0.77 (compensa ratio 1:3.4), gamma=2.0              │
    │ - Dice Loss: smooth=1.0                                                 │
    │ - Combinación: 0.6·Dice + 0.4·Focal (ponderada)                         │
    │ - Recomendado para: datasets con desbalance moderado-alto               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ USE_CLASS_BALANCING = False (Configuración ESTÁNDAR)                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ - Focal Loss: alpha=None (sin ponderación), gamma=2.0                   │
    │ - Dice Loss: smooth=1.0                                                 │
    │ - Combinación: Dice + Focal (suma simple, sin ponderación)              │
    │ - Recomendado para: ablation studies, comparaciones baseline            │
    └─────────────────────────────────────────────────────────────────────────┘

    Soporta deep supervision automáticamente.

    Returns:
        Función de pérdida compatible con deep supervision
    """
    # Dice Loss (igual en ambos modos)
    dice_loss = smp.losses.DiceLoss(
        mode='binary',
        smooth=Config.DICE_SMOOTH
    )

    if Config.USE_CLASS_BALANCING:
        # ═══════════════════════════════════════════════════════════════════
        # MODO CON COMPENSACIÓN EXPLÍCITA DE DESBALANCE
        # ═══════════════════════════════════════════════════════════════════
        focal_loss = smp.losses.FocalLoss(
            mode='binary',
            alpha=Config.FOCAL_ALPHA,   # Compensación explícita (0.77)
            gamma=Config.FOCAL_GAMMA    # Down-weight ejemplos fáciles (2.0)
        )
        w_dice = Config.LOSS_WEIGHT_DICE    # 0.6
        w_focal = Config.LOSS_WEIGHT_FOCAL  # 0.4
    else:
        # ═══════════════════════════════════════════════════════════════════
        # MODO ESTÁNDAR (SIN COMPENSACIÓN EXPLÍCITA)
        # ═══════════════════════════════════════════════════════════════════
        focal_loss = smp.losses.FocalLoss(
            mode='binary',
            alpha=None,                 # Sin ponderación de clases
            gamma=Config.FOCAL_GAMMA    # Mantiene down-weight ejemplos fáciles
        )
        w_dice = 1.0   # Suma simple
        w_focal = 1.0  # Suma simple

    def combined_loss(pred, target):
        """
        Calcula pérdida combinada.

        - Con balanceo: w_dice·Dice + w_focal·Focal (ponderada)
        - Sin balanceo: Dice + Focal (suma simple)

        Args:
            pred: Predicción del modelo
                  - Tensor [B, C, H, W] si deep_supervision=False
                  - Lista de tensores si deep_supervision=True (UNet++)
                  - Tupla (main, aux) si SegFormer con aux_loss=True
            target: Ground truth [B, C, H, W]

        Returns:
            Pérdida escalar
        """
        # Caso 1: Deep Supervision UNet++ (pred es una lista de outputs)
        if isinstance(pred, list):
            # Ponderación decreciente: salida final tiene más peso
            weights = [0.5, 0.25, 0.15, 0.1][:len(pred)]
            weights = [w / sum(weights) for w in weights]  # Normalizar

            total_loss = 0.0
            for pred_i, weight_i in zip(pred, weights):
                loss_i = w_dice * dice_loss(pred_i, target) + w_focal * focal_loss(pred_i, target)
                total_loss += weight_i * loss_i

            return total_loss

        # Caso 2: SegFormer con salida auxiliar (pred es una tupla)
        elif isinstance(pred, tuple) and len(pred) == 2:
            main_pred, aux_pred = pred
            # Pérdida principal
            main_loss = w_dice * dice_loss(main_pred, target) + w_focal * focal_loss(main_pred, target)
            # Pérdida auxiliar
            aux_loss = w_dice * dice_loss(aux_pred, target) + w_focal * focal_loss(aux_pred, target)
            # Combinar con peso auxiliar
            aux_weight = Config.SEGFORMER_AUX_WEIGHT if Config.MODEL_TYPE == "segformer" else 0.4
            return (1 - aux_weight) * main_loss + aux_weight * aux_loss

        # Caso 3: Sin Deep Supervision (pred es un tensor único)
        else:
            return w_dice * dice_loss(pred, target) + w_focal * focal_loss(pred, target)

    return combined_loss


def create_metrics():
    """
    Crea y retorna el diccionario de métricas.

    Las métricas se envuelven automáticamente para soportar:
    - Deep supervision (UNet++): pred es una lista
    - SegFormer con aux_loss: pred es una tupla (main, aux)
    - Modelos estándar: pred es un tensor

    Incluye métricas IoU por clase para análisis detallado de manglar disperso.

    Returns:
        Dict con métricas compatibles con todos los tipos de salida
    """

    def _extract_main_pred(pred):
        """
        Extrae la predicción principal de diferentes formatos de salida.

        Args:
            pred: Predicción del modelo (tensor, lista o tupla)

        Returns:
            Tensor de predicción principal
        """
        # Caso 1: Lista (deep supervision UNet++)
        if isinstance(pred, list):
            return pred[0]  # Output final (mayor resolución)

        # Caso 2: Tupla (SegFormer con aux_loss)
        elif isinstance(pred, tuple) and len(pred) == 2:
            return pred[0]  # Predicción principal

        # Caso 3: Tensor directo
        else:
            return pred

    def _make_metric_compatible(metric_fn):
        """
        Wrapper que hace una métrica compatible con todos los tipos de salida.

        Args:
            metric_fn: Función de métrica original

        Returns:
            Función de métrica modificada
        """
        def wrapped_metric(pred, target):
            main_pred = _extract_main_pred(pred)
            return metric_fn(main_pred, target)

        return wrapped_metric

    def iou_class_0(pred, target):
        """
        IoU para clase 0 (no-manglar / background).

        Útil para evaluar false positives en áreas sin manglar.
        """
        main_pred = _extract_main_pred(pred)
        # Invertir predicción y target para calcular IoU de clase 0
        pred_binary = (torch.sigmoid(main_pred) > 0.5).float()
        pred_inverted = 1 - pred_binary
        target_inverted = 1 - target

        # Calcular intersección y unión
        intersection = (pred_inverted * target_inverted).sum()
        union = pred_inverted.sum() + target_inverted.sum() - intersection
        iou_0 = (intersection + 1e-6) / (union + 1e-6)
        return iou_0

    def iou_class_1(pred, target):
        """
        IoU para clase 1 (manglar).

        Métrica principal para evaluar detección de manglar,
        especialmente importante en teselas positive_sparse.
        """
        main_pred = _extract_main_pred(pred)
        # Calcular IoU directamente para clase 1 (manglar)
        pred_binary = (torch.sigmoid(main_pred) > 0.5).float()

        # Calcular intersección y unión
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou_1 = (intersection + 1e-6) / (union + 1e-6)
        return iou_1

    def mean_iou(pred, target):
        """
        Mean IoU (promedio de IoU por clase).

        mIoU = (IoU_class0 + IoU_class1) / 2

        Esta métrica da igual peso a ambas clases, lo cual puede ser
        más informativo que el IoU global en casos de desbalance.
        """
        iou_0 = iou_class_0(pred, target)
        iou_1 = iou_class_1(pred, target)
        return (iou_0 + iou_1) / 2

    # Envolver todas las métricas
    return {
        'OA': _make_metric_compatible(overall_accuracy),
        'iou': _make_metric_compatible(iou),  # IoU global (igual al original)
        'iou_0': iou_class_0,  # IoU clase 0 (no-manglar) - NUEVO
        'iou_1': iou_class_1,  # IoU clase 1 (manglar) - NUEVO
        'mIoU': mean_iou,  # Mean IoU - NUEVO
        'f1': _make_metric_compatible(f1_score),
        'pre': _make_metric_compatible(precision),
        'recall': _make_metric_compatible(recall),
        'FWIoU': _make_metric_compatible(fw_iou),
        'kp': _make_metric_compatible(kappa),
    }


# ============================================================================
# CALLBACKS
# ============================================================================

def create_callbacks():
    """Crea y retorna los callbacks de Lightning."""

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=Config.CHECKPOINT_MONITOR,
        mode='max',
        save_top_k=1,
        dirpath=Config.CHECKPOINT_DIR,
        filename=Config.RUN_NAME + '-{epoch:02d}-{val_iou:.4f}',
        verbose=True,
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=Config.CHECKPOINT_MONITOR,
        mode='max',
        patience=Config.EARLY_STOPPING_PATIENCE,
        verbose=True,
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Loss history callback (para generar figuras de convergencia)
    loss_history = LossHistoryCallback()

    # Metrics history callback (para registro completo de métricas)
    metrics_history = MetricsHistoryCallback(
        output_dir=Config.STATS_DIR,
        experiment_name=Config.RUN_NAME,
        config=Config
    )

    return [checkpoint_callback, early_stopping, lr_monitor, loss_history, metrics_history]


# ============================================================================
# LIGHTNING MODULE
# ============================================================================

def create_lightning_module(model, optimizer, scheduler, loss_fn, metrics):
    """Crea y retorna el LightningModule."""

    # Configuración del scheduler para Lightning
    if Config.WARMUP_EPOCHS > 0:
        # Para LambdaLR (warmup), no necesitamos monitor
        scheduler_config = {
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        # Para ReduceLROnPlateau, necesitamos monitor
        scheduler_config = {
            'monitor': Config.SCHEDULER_MONITOR,
            'interval': 'epoch',
            'frequency': 1,
        }

    module = Module(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    return module


# ============================================================================
# TRAINER
# ============================================================================

def create_trainer(callbacks):
    """Crea y retorna el Trainer de Lightning."""
    trainer = L.Trainer(
        max_epochs=Config.EPOCHS,
        accelerator='gpu',
        devices=1,
        precision=Config.PRECISION,
        logger=TensorBoardLogger(
            save_dir=Config.LOG_DIR,
            name=Config.EXPERIMENT_NAME,
            version=Config.RUN_NAME,
        ),
        callbacks=callbacks,
        log_every_n_steps=Config.LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=1,
    )
    return trainer


# ============================================================================
# ANÁLISIS POST-ENTRENAMIENTO
# ============================================================================

def analyze_checkpoints():
    """Analiza los checkpoints guardados y encuentra el mejor."""
    import glob
    import re
    import json
    
    print("\n" + "="*80)
    print("ANÁLISIS DE CHECKPOINTS")
    print("="*80)
    
    # Buscar checkpoints
    pattern = f'{Config.CHECKPOINT_DIR}/{Config.RUN_NAME}*.ckpt'
    checkpoints = sorted(glob.glob(pattern))
    
    if not checkpoints:
        print("No se encontraron checkpoints.")
        return
    
    print(f"Total de checkpoints encontrados: {len(checkpoints)}\n")
    
    def parse_iou(checkpoint_path):
        """Extrae val_iou del nombre del checkpoint."""
        match = re.search(r'val_iou=(\d+\.\d+)', checkpoint_path)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    # Parsear información de cada checkpoint
    ckpt_info = []
    
    for ckpt in checkpoints:
        epoch_match = re.search(r'epoch=(\d+)', ckpt)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        
        val_iou = parse_iou(ckpt)
        
        if val_iou is not None:
            ckpt_info.append((epoch, val_iou, ckpt))
            print(f"✓ Epoch {epoch:2d} - val_iou: {val_iou:.4f} - {Path(ckpt).name}")
    
    if not ckpt_info:
        print("No se pudo extraer información de los checkpoints.")
        return
    
    # Encontrar el mejor
    ckpt_info.sort(key=lambda x: x[1], reverse=True)
    best_epoch, best_iou, best_ckpt = ckpt_info[0]
    
    print("\n" + "="*80)
    print("MEJOR MODELO")
    print("="*80)
    print(f"Epoch: {best_epoch}")
    print(f"Val IoU: {best_iou:.4f}")
    print(f"Archivo: {Path(best_ckpt).name}")
    print(f"Path completo: {Path(best_ckpt).absolute()}")
    print("="*80)
    
    # Guardar información
    best_model_info = {
        'epoch': best_epoch,
        'val_iou': best_iou,
        'checkpoint_path': str(Path(best_ckpt).absolute()),
        'filename': Path(best_ckpt).name,
        'total_checkpoints': len(ckpt_info)
    }
    
    info_file = Path(Config.CHECKPOINT_DIR) / 'best_model_info.json'
    with open(info_file, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"\n✓ Información guardada en: {info_file}")
    print("="*80 + "\n")

    return best_ckpt  # Retornar path del mejor checkpoint


# ============================================================================
# MATRIZ DE CONFUSIÓN - ANÁLISIS POST-ENTRENAMIENTO
# ============================================================================

def compute_confusion_matrix_validation(
    checkpoint_path: str,
    dm: 'TorchGeoDataModule',
    model_fn: callable,
    device: str = 'cuda'
) -> dict:
    """
    Calcula la matriz de confusión a nivel de píxel sobre el conjunto de validación.

    Esta función carga el mejor modelo entrenado, realiza inferencia sobre todas
    las teselas de validación, y computa la matriz de confusión agregada.

    Args:
        checkpoint_path: Ruta al checkpoint del mejor modelo (.ckpt)
        dm: DataModule configurado con datos de validación
        model_fn: Función para crear el modelo base
        device: Dispositivo para inferencia ('cuda' o 'cpu')

    Returns:
        Dict con matriz de confusión y métricas derivadas:
            - confusion_matrix: np.ndarray (2x2)
            - tn, fp, fn, tp: Valores individuales
            - total_pixels: Total de píxeles evaluados
            - accuracy, precision, recall, f1, iou: Métricas derivadas
    """
    import glob

    print("\n" + "="*80)
    print("CÁLCULO DE MATRIZ DE CONFUSIÓN - CONJUNTO DE VALIDACIÓN")
    print("="*80)

    # Verificar checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint no encontrado: {checkpoint_path}")
        return None

    print(f"📂 Cargando modelo desde: {Path(checkpoint_path).name}")

    # Cargar modelo
    model = model_fn()

    # Crear LightningModule temporal para cargar pesos
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    loss_fn = create_loss_fn()
    metrics = create_metrics()

    module = create_lightning_module(model, optimizer, scheduler, loss_fn, metrics)

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    module.load_state_dict(checkpoint['state_dict'])
    module.eval()
    module.to(device)

    print(f"✓ Modelo cargado correctamente")

    # Preparar datamodule
    if not hasattr(dm, 'val_ds'):
        dm.setup('fit')

    val_dataset = dm.val_ds
    n_samples = len(val_dataset)
    print(f"📊 Procesando {n_samples} teselas de validación...")

    # Acumuladores para matriz de confusión
    all_preds = []
    all_targets = []

    # Inferencia
    with torch.no_grad():
        for idx in range(n_samples):
            sample = val_dataset[idx]

            # Extraer imagen y máscara
            if isinstance(sample, dict):
                image = sample['image']
                mask = sample['mask']
            else:
                image, mask = sample

            # Preparar tensor de entrada
            if not torch.is_tensor(image):
                image = torch.tensor(image)
            image = image.unsqueeze(0).float().to(device)

            # Preparar máscara de referencia
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            while mask.ndim > 2:
                mask = mask.squeeze()

            # Predicción
            logits = module(image)

            # Manejar deep supervision
            if isinstance(logits, list):
                logits = logits[0]

            # Convertir a predicción binaria
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()
            else:
                pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            # Aplanar y acumular
            all_preds.extend(pred.flatten().astype(int))
            all_targets.extend(mask.flatten().astype(int))

            # Progreso
            if (idx + 1) % 20 == 0:
                print(f"   Procesadas {idx + 1}/{n_samples} teselas...")

    print(f"✓ Inferencia completada: {len(all_targets):,} píxeles evaluados")

    # Calcular matriz de confusión
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Extraer valores
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    # Calcular métricas derivadas
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Resultados
    results = {
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total_pixels': int(total),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'iou': float(iou)
    }

    # Imprimir resultados
    print("\n" + "-"*80)
    print("MATRIZ DE CONFUSIÓN (píxeles)")
    print("-"*80)
    print(f"\n                    Predicción")
    print(f"                 No-Manglar   Manglar")
    print(f"Real No-Manglar    {tn:>10,}   {fp:>10,}   (TN, FP)")
    print(f"Real Manglar       {fn:>10,}   {tp:>10,}   (FN, TP)")
    print(f"\n📊 Total de píxeles evaluados: {total:,}")
    print(f"\n📈 Métricas derivadas:")
    print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"   Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"   F1-Score:    {f1:.4f}")
    print(f"   IoU:         {iou:.4f}")
    print(f"\n⚠️  Errores de clasificación:")
    print(f"   Falsos Positivos (FP): {fp:,} píxeles ({fp/total*100:.3f}%)")
    print(f"   Falsos Negativos (FN): {fn:,} píxeles ({fn/total*100:.3f}%)")
    print("-"*80)

    return results


def compute_confusion_matrix_macro(
    checkpoint_path: str,
    dm: 'TorchGeoDataModule',
    model_fn: callable,
    device: str = 'cuda'
) -> dict:
    """
    Calcula la matriz de confusión MACRO-PROMEDIADA sobre el conjunto de validación.

    Esta función calcula IoU por tesela y luego promedia, dando el mismo peso a cada tesela
    independientemente de su tamaño o proporción de manglar. Esto permite identificar
    teselas espacialmente desafiantes (manglar disperso, zonas de transición).

    DIFERENCIAS CON compute_confusion_matrix_validation():
    - compute_confusion_matrix_validation: Agrega TODOS los píxeles primero, luego calcula métricas
      → IoU micro-promediada (refleja rendimiento a nivel de píxel, mayor peso a áreas grandes)
    - compute_confusion_matrix_macro: Calcula métricas POR TESELA, luego promedia
      → IoU macro-promediada (cada tesela tiene mismo peso, identifica desafíos espaciales)

    Args:
        checkpoint_path: Ruta al checkpoint del mejor modelo (.ckpt)
        dm: DataModule configurado con datos de validación
        model_fn: Función para crear el modelo base
        device: Dispositivo para inferencia ('cuda' o 'cpu')

    Returns:
        Dict con matriz de confusión macro y métricas derivadas:
            - confusion_matrix_macro: np.ndarray (2x2) promediada por tesela
            - tn_macro, fp_macro, fn_macro, tp_macro: Valores promedio por tesela
            - accuracy_macro, precision_macro, recall_macro, f1_macro, iou_macro: Métricas macro
            - per_tile_ious: Lista de IoUs individuales por tesela
            - per_tile_metrics: Lista de métricas por tesela
    """
    import glob

    print("\n" + "="*80)
    print("CÁLCULO DE MATRIZ DE CONFUSIÓN - MACRO-PROMEDIADA (POR TESELA)")
    print("="*80)

    # Verificar checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint no encontrado: {checkpoint_path}")
        return None

    print(f"📂 Cargando modelo desde: {Path(checkpoint_path).name}")

    # Cargar modelo
    model = model_fn()

    # Crear LightningModule temporal para cargar pesos
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    loss_fn = create_loss_fn()
    metrics = create_metrics()

    module = create_lightning_module(model, optimizer, scheduler, loss_fn, metrics)

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    module.load_state_dict(checkpoint['state_dict'])
    module.eval()
    module.to(device)

    print(f"✓ Modelo cargado correctamente")

    # Preparar datamodule
    if not hasattr(dm, 'val_ds'):
        dm.setup('fit')

    val_dataset = dm.val_ds
    n_samples = len(val_dataset)
    print(f"📊 Procesando {n_samples} teselas de validación...")
    print(f"📍 Calculando IoU por tesela (macro-averaging)...")

    # Listas para almacenar métricas por tesela
    per_tile_ious = []
    per_tile_metrics = []
    tile_confusion_matrices = []

    # Inferencia por tesela
    with torch.no_grad():
        for idx in range(n_samples):
            sample = val_dataset[idx]

            # Extraer imagen y máscara
            if isinstance(sample, dict):
                image = sample['image']
                mask = sample['mask']
            else:
                image, mask = sample

            # Preparar tensor de entrada
            if not torch.is_tensor(image):
                image = torch.tensor(image)
            image = image.unsqueeze(0).float().to(device)

            # Preparar máscara de referencia
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            while mask.ndim > 2:
                mask = mask.squeeze()

            # Predicción
            logits = module(image)

            # Manejar deep supervision
            if isinstance(logits, list):
                logits = logits[0]

            # Convertir a predicción binaria
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()
            else:
                pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            # Calcular matriz de confusión para esta tesela
            y_true_tile = mask.flatten().astype(int)
            y_pred_tile = pred.flatten().astype(int)

            cm_tile = confusion_matrix(y_true_tile, y_pred_tile, labels=[0, 1])
            tile_confusion_matrices.append(cm_tile)

            # Extraer valores para esta tesela
            tn_tile, fp_tile, fn_tile, tp_tile = cm_tile.ravel()

            # Calcular métricas para esta tesela
            accuracy_tile = (tp_tile + tn_tile) / (tn_tile + fp_tile + fn_tile + tp_tile)
            precision_tile = tp_tile / (tp_tile + fp_tile) if (tp_tile + fp_tile) > 0 else 0
            recall_tile = tp_tile / (tp_tile + fn_tile) if (tp_tile + fn_tile) > 0 else 0
            f1_tile = 2 * precision_tile * recall_tile / (precision_tile + recall_tile) if (precision_tile + recall_tile) > 0 else 0
            iou_tile = tp_tile / (tp_tile + fp_tile + fn_tile) if (tp_tile + fp_tile + fn_tile) > 0 else 0
            specificity_tile = tn_tile / (tn_tile + fp_tile) if (tn_tile + fp_tile) > 0 else 0

            per_tile_ious.append(iou_tile)
            per_tile_metrics.append({
                'tile_idx': idx,
                'tn': int(tn_tile),
                'fp': int(fp_tile),
                'fn': int(fn_tile),
                'tp': int(tp_tile),
                'accuracy': float(accuracy_tile),
                'precision': float(precision_tile),
                'recall': float(recall_tile),
                'specificity': float(specificity_tile),
                'f1': float(f1_tile),
                'iou': float(iou_tile)
            })

            # Progreso
            if (idx + 1) % 20 == 0:
                print(f"   Procesadas {idx + 1}/{n_samples} teselas...")

    print(f"✓ Inferencia completada: {n_samples} teselas evaluadas")

    # ========================================================================
    # CLASIFICACIÓN DE TESELAS (MACRO A NIVEL DE TESELA)
    # ========================================================================
    # Umbral de IoU para considerar una tesela como "bien clasificada"
    IOU_THRESHOLD = 0.7

    # Umbral de porcentaje de manglar para considerar que una tesela "tiene manglar"
    MANGROVE_THRESHOLD = 0.05  # 5% de píxeles deben ser manglar

    # Contadores de teselas
    tp_tiles = 0  # Teselas con manglar correctamente clasificadas (IoU >= umbral)
    fn_tiles = 0  # Teselas con manglar mal clasificadas (IoU < umbral)
    tn_tiles = 0  # Teselas sin manglar correctamente clasificadas
    fp_tiles = 0  # Teselas sin manglar con falsos positivos significativos

    # Clasificar cada tesela
    for tile_metrics in per_tile_metrics:
        tp_px = tile_metrics['tp']
        fn_px = tile_metrics['fn']
        fp_px = tile_metrics['fp']
        total_px = 16384  # 128x128
        iou = tile_metrics['iou']

        # Determinar si la tesela tiene manglar real
        mangrove_pixels = tp_px + fn_px
        has_mangrove = (mangrove_pixels / total_px) >= MANGROVE_THRESHOLD

        if has_mangrove:
            # Tesela con manglar real
            if iou >= IOU_THRESHOLD:
                tp_tiles += 1  # Bien clasificada
            else:
                fn_tiles += 1  # Mal clasificada
        else:
            # Tesela sin manglar real (o muy poco)
            fp_rate = fp_px / total_px
            if fp_rate < 0.05:  # Menos de 5% de falsos positivos
                tn_tiles += 1  # Correctamente clasificada sin manglar
            else:
                fp_tiles += 1  # Falsos positivos significativos

    # Crear matriz de confusión a nivel de tesela
    cm_tiles = np.array([[tn_tiles, fp_tiles],
                         [fn_tiles, tp_tiles]])

    # Calcular métricas a nivel de tesela
    total_tiles = n_samples
    accuracy_tiles = (tp_tiles + tn_tiles) / total_tiles
    precision_tiles = tp_tiles / (tp_tiles + fp_tiles) if (tp_tiles + fp_tiles) > 0 else 0
    recall_tiles = tp_tiles / (tp_tiles + fn_tiles) if (tp_tiles + fn_tiles) > 0 else 0
    specificity_tiles = tn_tiles / (tn_tiles + fp_tiles) if (tn_tiles + fp_tiles) > 0 else 0
    f1_tiles = 2 * precision_tiles * recall_tiles / (precision_tiles + recall_tiles) if (precision_tiles + recall_tiles) > 0 else 0

    # Calcular IoU promedio (se mantiene como antes)
    iou_macro = np.mean(per_tile_ious)
    iou_std = np.std(per_tile_ious)
    iou_min = np.min(per_tile_ious)
    iou_max = np.max(per_tile_ious)
    iou_median = np.median(per_tile_ious)

    # Resultados
    results = {
        'confusion_matrix_tiles': cm_tiles,
        'tn_tiles': int(tn_tiles),
        'fp_tiles': int(fp_tiles),
        'fn_tiles': int(fn_tiles),
        'tp_tiles': int(tp_tiles),
        'n_tiles': n_samples,
        'iou_threshold': IOU_THRESHOLD,
        'accuracy_tiles': float(accuracy_tiles),
        'precision_tiles': float(precision_tiles),
        'recall_tiles': float(recall_tiles),
        'specificity_tiles': float(specificity_tiles),
        'f1_tiles': float(f1_tiles),
        'iou_macro': float(iou_macro),
        'iou_std': float(iou_std),
        'iou_min': float(iou_min),
        'iou_max': float(iou_max),
        'iou_median': float(iou_median),
        'per_tile_ious': per_tile_ious,
        'per_tile_metrics': per_tile_metrics
    }

    # Imprimir resultados
    print("\n" + "-"*80)
    print("MATRIZ DE CONFUSIÓN MACRO - CLASIFICACIÓN POR TESELA")
    print("-"*80)
    print(f"\n⚙️  Umbral de IoU para clasificación: {IOU_THRESHOLD:.2f}")
    print(f"    (Teselas con IoU >= {IOU_THRESHOLD:.2f} se consideran 'bien clasificadas')")
    print(f"\n                    Predicción")
    print(f"                 No-Manglar   Manglar")
    print(f"Real No-Manglar    {tn_tiles:>10d}   {fp_tiles:>10d}   (TN, FP) teselas")
    print(f"Real Manglar       {fn_tiles:>10d}   {tp_tiles:>10d}   (FN, TP) teselas")
    print(f"\n📊 Total de teselas: {n_samples}")
    print(f"\n📈 Métricas de clasificación de teselas:")
    print(f"   Accuracy:    {accuracy_tiles:.4f} ({accuracy_tiles*100:.2f}%)")
    print(f"   Precision:   {precision_tiles:.4f} ({precision_tiles*100:.2f}%)")
    print(f"   Recall:      {recall_tiles:.4f} ({recall_tiles*100:.2f}%)")
    print(f"   Specificity: {specificity_tiles:.4f} ({specificity_tiles*100:.2f}%)")
    print(f"   F1-Score:    {f1_tiles:.4f}")
    print(f"\n📊 Distribución de IoU por tesela:")
    print(f"   Media:   {iou_macro:.4f}")
    print(f"   Mediana: {iou_median:.4f}")
    print(f"   Std Dev: {iou_std:.4f}")
    print(f"   Min:     {iou_min:.4f}")
    print(f"   Max:     {iou_max:.4f}")
    print(f"\n💡 Interpretación:")
    print(f"   • TP: {tp_tiles} teselas con manglar correctamente clasificadas (IoU ≥ {IOU_THRESHOLD:.2f})")
    print(f"   • FN: {fn_tiles} teselas con manglar mal clasificadas (IoU < {IOU_THRESHOLD:.2f})")
    print(f"   • TN: {tn_tiles} teselas sin manglar correctamente clasificadas")
    print(f"   • FP: {fp_tiles} teselas sin manglar con falsos positivos significativos")
    print("-"*80)

    return results


def plot_confusion_matrix_macro(
    cm_results_macro: dict,
    model_name: str = "Multi-Branch UNet++",
    save_path: str = None,
    figsize: tuple = (10, 8),
    dpi: int = 300
) -> plt.Figure:
    """
    Genera una figura de matriz de confusión MACRO con clasificación por tesela.

    Esta visualización muestra cuántas teselas completas son clasificadas correctamente
    basándose en un umbral de IoU. Cada tesela tiene el mismo peso, revelando el rendimiento
    del modelo en regiones espacialmente desafiantes.

    Args:
        cm_results_macro: Dict retornado por compute_confusion_matrix_macro()
        model_name: Nombre del modelo para el título
        save_path: Ruta donde guardar la figura (opcional)
        figsize: Tamaño de la figura en pulgadas
        dpi: Resolución de la imagen (≥300 para publicación)

    Returns:
        plt.Figure: Figura de matplotlib generada
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    # Extraer matriz de confusión de teselas
    cm = cm_results_macro['confusion_matrix_tiles']
    tn, fp, fn, tp = cm_results_macro['tn_tiles'], cm_results_macro['fp_tiles'], cm_results_macro['fn_tiles'], cm_results_macro['tp_tiles']
    n_tiles = cm_results_macro['n_tiles']
    iou_threshold = cm_results_macro['iou_threshold']

    # Crear figura con estilo profesional
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Colormap personalizado (azul-blanco-rojo no es ideal, usar blues para TN/TP)
    cmap = plt.cm.Oranges  # Diferente colormap para distinguir de micro

    # Normalizar para el colormap
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot de la matriz
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proporción de Teselas', rotation=-90, va="bottom", fontsize=12)

    # Etiquetas
    classes = ['No-Manglar\n(Clase 0)', 'Manglar\n(Clase 1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Anotaciones con conteo de teselas
    thresh = cm_normalized.max() / 2.
    total_tiles = cm.sum()

    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            proportion = cm_normalized[i, j]
            pct_of_total = (value / total_tiles) * 100

            # Etiqueta según posición con conteo de teselas
            if i == 0 and j == 0:
                label = f"TN\n{value} teselas\n({pct_of_total:.1f}%)"
            elif i == 0 and j == 1:
                label = f"FP\n{value} teselas\n({pct_of_total:.1f}%)"
            elif i == 1 and j == 0:
                label = f"FN\n{value} teselas\n({pct_of_total:.1f}%)"
            else:
                label = f"TP\n{value} teselas\n({pct_of_total:.1f}%)"

            color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(j, i, label, ha="center", va="center", color=color,
                   fontsize=13, fontweight='bold')

    # Títulos y etiquetas
    ax.set_xlabel('Predicción', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=14, fontweight='bold')
    ax.set_title(f'Matriz de Confusión MACRO - {model_name}\n'
                 f'(Clasificación de {n_tiles} teselas - IoU umbral ≥ {iou_threshold:.2f})\n'
                 f'Cada tesela clasificada como unidad completa',
                 fontsize=12, fontweight='bold', pad=20)

    # Añadir métricas como texto
    metrics_text = (
        f"IoU Macro: {cm_results_macro['iou_macro']:.4f}\n"
        f"IoU Min:   {cm_results_macro['iou_min']:.4f}\n"
        f"IoU Max:   {cm_results_macro['iou_max']:.4f}\n"
        f"IoU Std:   {cm_results_macro['iou_std']:.4f}\n"
        f"F1-Score:  {cm_results_macro['f1_tiles']:.4f}\n"
        f"Accuracy:  {cm_results_macro['accuracy_tiles']:.4f}"
    )

    # Caja de texto con métricas
    props = dict(boxstyle='round', facecolor='bisque', alpha=0.8)
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props, family='monospace')

    # Añadir nota explicativa
    note_text = (
        f"Nota: Teselas con manglar se clasifican como TP si IoU ≥ {iou_threshold:.2f},\n"
        f"como FN si IoU < {iou_threshold:.2f}. Revela rendimiento en casos desafiantes\n"
        "(manglar disperso, bordes, transiciones intermareales)"
    )
    ax.text(0.5, -0.15, note_text, transform=ax.transAxes, fontsize=9,
            ha='center', style='italic', color='gray')

    plt.tight_layout()

    # Guardar figura
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Figura macro guardada en: {save_path}")

    return fig


def plot_confusion_matrix(
    cm_results: dict,
    model_name: str = "Multi-Branch UNet++",
    save_path: str = None,
    figsize: tuple = (10, 8),
    dpi: int = 300
) -> plt.Figure:
    """
    Genera una figura de matriz de confusión con calidad de publicación.

    Crea una visualización profesional de la matriz de confusión con
    anotaciones de valores absolutos y porcentajes, adecuada para
    inclusión directa en tesis y artículos científicos.

    Args:
        cm_results: Dict retornado por compute_confusion_matrix_validation()
        model_name: Nombre del modelo para el título
        save_path: Ruta donde guardar la figura (opcional)
        figsize: Tamaño de la figura en pulgadas
        dpi: Resolución de la imagen (≥300 para publicación)

    Returns:
        plt.Figure: Figura de matplotlib generada
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    cm = cm_results['confusion_matrix']
    tn, fp, fn, tp = cm_results['tn'], cm_results['fp'], cm_results['fn'], cm_results['tp']
    total = cm_results['total_pixels']

    # Crear figura con estilo profesional
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Colormap personalizado (azul-blanco-rojo no es ideal, usar blues para TN/TP)
    cmap = plt.cm.Blues

    # Normalizar para el colormap
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot de la matriz
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proporción', rotation=-90, va="bottom", fontsize=12)

    # Etiquetas
    classes = ['No-Manglar\n(Clase 0)', 'Manglar\n(Clase 1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Anotaciones con valores absolutos y porcentajes
    thresh = cm_normalized.max() / 2.
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            percentage = value / total * 100

            # Etiqueta según posición
            if i == 0 and j == 0:
                label = f"TN\n{value:,}\n({percentage:.2f}%)"
            elif i == 0 and j == 1:
                label = f"FP\n{value:,}\n({percentage:.2f}%)"
            elif i == 1 and j == 0:
                label = f"FN\n{value:,}\n({percentage:.2f}%)"
            else:
                label = f"TP\n{value:,}\n({percentage:.2f}%)"

            color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(j, i, label, ha="center", va="center", color=color,
                   fontsize=14, fontweight='bold')

    # Títulos y etiquetas
    ax.set_xlabel('Predicción', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=14, fontweight='bold')
    ax.set_title(f'Matriz de Confusión - {model_name}\n'
                 f'(n = {total:,} píxeles, 80 teselas de validación)',
                 fontsize=14, fontweight='bold', pad=20)

    # Añadir métricas como texto
    metrics_text = (
        f"Accuracy: {cm_results['accuracy']:.4f}\n"
        f"Precision: {cm_results['precision']:.4f}\n"
        f"Recall: {cm_results['recall']:.4f}\n"
        f"F1-Score: {cm_results['f1']:.4f}\n"
        f"IoU: {cm_results['iou']:.4f}"
    )

    # Caja de texto con métricas
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props, family='monospace')

    plt.tight_layout()

    # Guardar figura
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Figura guardada en: {save_path}")

    return fig


def export_confusion_matrix_results(
    cm_results: dict,
    model_name: str,
    output_dir: str = None
) -> dict:
    """
    Exporta los resultados de la matriz de confusión a archivos.

    Guarda:
    - Archivo JSON con todos los valores numéricos
    - Archivo TXT con formato legible

    Args:
        cm_results: Dict retornado por compute_confusion_matrix_validation()
        model_name: Nombre del modelo
        output_dir: Directorio de salida (default: Config.STATS_DIR)

    Returns:
        Dict con rutas de archivos generados
    """
    if output_dir is None:
        output_dir = Config.STATS_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Nombre base
    base_name = model_name.lower().replace(' ', '_').replace('-', '_').replace('+', 'plus')

    # Preparar datos para JSON (convertir numpy a tipos nativos)
    json_data = {
        'model_name': model_name,
        'validation_tiles': 80,
        'tile_size': f"{Config.TILE_SIZE}x{Config.TILE_SIZE}",
        'total_pixels': cm_results['total_pixels'],
        'confusion_matrix': {
            'true_negative': cm_results['tn'],
            'false_positive': cm_results['fp'],
            'false_negative': cm_results['fn'],
            'true_positive': cm_results['tp']
        },
        'metrics': {
            'accuracy': cm_results['accuracy'],
            'precision': cm_results['precision'],
            'recall': cm_results['recall'],
            'specificity': cm_results['specificity'],
            'f1_score': cm_results['f1'],
            'iou': cm_results['iou']
        },
        'error_analysis': {
            'fp_count': cm_results['fp'],
            'fp_percentage': cm_results['fp'] / cm_results['total_pixels'] * 100,
            'fn_count': cm_results['fn'],
            'fn_percentage': cm_results['fn'] / cm_results['total_pixels'] * 100,
            'total_errors': cm_results['fp'] + cm_results['fn'],
            'error_rate': (cm_results['fp'] + cm_results['fn']) / cm_results['total_pixels'] * 100
        }
    }

    # Guardar JSON
    json_path = Path(output_dir) / f'{base_name}_confusion_matrix.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Guardar TXT
    txt_path = Path(output_dir) / f'{base_name}_confusion_matrix.txt'
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MATRIZ DE CONFUSIÓN - {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Conjunto de validación: 80 teselas ({Config.TILE_SIZE}x{Config.TILE_SIZE} píxeles)\n")
        f.write(f"Total de píxeles evaluados: {cm_results['total_pixels']:,}\n\n")
        f.write("-"*80 + "\n")
        f.write("MATRIZ DE CONFUSIÓN\n")
        f.write("-"*80 + "\n\n")
        f.write("                      Predicción\n")
        f.write("                   No-Manglar    Manglar\n")
        f.write(f"Real No-Manglar    {cm_results['tn']:>10,}  {cm_results['fp']:>10,}   (TN, FP)\n")
        f.write(f"Real Manglar       {cm_results['fn']:>10,}  {cm_results['tp']:>10,}   (FN, TP)\n\n")
        f.write("-"*80 + "\n")
        f.write("MÉTRICAS DERIVADAS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Accuracy:     {cm_results['accuracy']:.4f} ({cm_results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:    {cm_results['precision']:.4f} ({cm_results['precision']*100:.2f}%)\n")
        f.write(f"Recall:       {cm_results['recall']:.4f} ({cm_results['recall']*100:.2f}%)\n")
        f.write(f"Specificity:  {cm_results['specificity']:.4f} ({cm_results['specificity']*100:.2f}%)\n")
        f.write(f"F1-Score:     {cm_results['f1']:.4f}\n")
        f.write(f"IoU:          {cm_results['iou']:.4f}\n\n")
        f.write("-"*80 + "\n")
        f.write("ANÁLISIS DE ERRORES\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Falsos Positivos (FP): {cm_results['fp']:,} píxeles ({cm_results['fp']/cm_results['total_pixels']*100:.3f}%)\n")
        f.write(f"  → No-manglar clasificado erróneamente como manglar\n\n")
        f.write(f"Falsos Negativos (FN): {cm_results['fn']:,} píxeles ({cm_results['fn']/cm_results['total_pixels']*100:.3f}%)\n")
        f.write(f"  → Manglar no detectado (omisión)\n\n")
        f.write(f"Total de errores: {cm_results['fp'] + cm_results['fn']:,} píxeles\n")
        f.write(f"Tasa de error:    {(cm_results['fp'] + cm_results['fn'])/cm_results['total_pixels']*100:.3f}%\n")
        f.write("\n" + "="*80 + "\n")

    print(f"✓ Resultados JSON guardados en: {json_path}")
    print(f"✓ Resultados TXT guardados en: {txt_path}")

    return {
        'json_path': str(json_path),
        'txt_path': str(txt_path)
    }


def run_confusion_matrix_analysis(dm: 'TorchGeoDataModule', model_fn: callable) -> dict:
    """
    Ejecuta el análisis completo de matriz de confusión (MICRO Y MACRO).

    Función wrapper que orquesta todo el proceso:
    1. Busca el mejor checkpoint
    2. Calcula matriz de confusión MICRO (agregación global de píxeles)
    3. Calcula matriz de confusión MACRO (promedio por tesela)
    4. Genera visualizaciones para ambas escalas
    5. Exporta resultados

    DIFERENCIAS ENTRE ESCALAS:
    - MICRO (IoU micro-promediada): Agrega TODOS los píxeles, luego calcula IoU
      → Refleja rendimiento operativo a nivel de píxel
      → Mayor IoU debido al peso de áreas extensas de manglar continuo

    - MACRO (IoU macro-promediada): Calcula IoU por tesela, luego promedia
      → Cada tesela tiene el mismo peso independiente de su tamaño
      → Revela rendimiento en regiones desafiantes (manglar disperso, bordes)
      → IoU más baja pero más representativa de desafíos espaciales

    Args:
        dm: DataModule configurado
        model_fn: Función para crear el modelo base

    Returns:
        Dict con resultados y rutas de archivos generados para ambas escalas
    """
    import glob

    print("\n" + "="*80)
    print("INICIANDO ANÁLISIS DE MATRIZ DE CONFUSIÓN (MICRO + MACRO)")
    print("="*80)

    # Buscar el mejor checkpoint
    pattern = f'{Config.CHECKPOINT_DIR}/{Config.RUN_NAME}*.ckpt'
    checkpoints = sorted(glob.glob(pattern))

    if not checkpoints:
        print("⚠️  No se encontraron checkpoints para analizar.")
        return None

    # Obtener el mejor (asumiendo que analyze_checkpoints ya se ejecutó)
    best_info_path = Path(Config.CHECKPOINT_DIR) / 'best_model_info.json'
    if best_info_path.exists():
        with open(best_info_path, 'r') as f:
            best_info = json.load(f)
        best_ckpt = best_info['checkpoint_path']
    else:
        # Usar el último checkpoint como fallback
        best_ckpt = checkpoints[-1]

    print(f"📂 Usando checkpoint: {Path(best_ckpt).name}")

    # Determinar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Dispositivo: {device}")

    # ========================================================================
    # ANÁLISIS MICRO (AGREGACIÓN GLOBAL DE PÍXELES)
    # ========================================================================
    print("\n" + "="*80)
    print("PASO 1: MATRIZ DE CONFUSIÓN MICRO (Agregación Global de Píxeles)")
    print("="*80)

    # Calcular matriz de confusión micro
    cm_results_micro = compute_confusion_matrix_validation(
        checkpoint_path=best_ckpt,
        dm=dm,
        model_fn=model_fn,
        device=device
    )

    if cm_results_micro is None:
        return None

    # Nombre del modelo para archivos
    model_name = Config.RUN_NAME.replace('-Sentinel2', '').replace('_', ' ')

    # Generar visualización MICRO
    fig_path_micro = Path(Config.FIGURES_DIR) / f'confusion_matrix_micro_{Config.MODEL_TYPE}.png'
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)

    plot_confusion_matrix(
        cm_results=cm_results_micro,
        model_name=model_name + " (Micro)",
        save_path=str(fig_path_micro),
        dpi=300
    )

    # Exportar resultados MICRO
    export_paths_micro = export_confusion_matrix_results(
        cm_results=cm_results_micro,
        model_name=model_name + " Micro",
        output_dir=Config.STATS_DIR
    )

    # Cerrar figura para liberar memoria
    plt.close('all')

    print("\n✓ Análisis MICRO completado")

    # ========================================================================
    # ANÁLISIS MACRO (PROMEDIO POR TESELA)
    # ========================================================================
    print("\n" + "="*80)
    print("PASO 2: MATRIZ DE CONFUSIÓN MACRO (Promedio por Tesela)")
    print("="*80)

    # Calcular matriz de confusión macro
    cm_results_macro = compute_confusion_matrix_macro(
        checkpoint_path=best_ckpt,
        dm=dm,
        model_fn=model_fn,
        device=device
    )

    if cm_results_macro is None:
        print("⚠️  No se pudo calcular la matriz macro")
        return {
            'cm_results_micro': cm_results_micro,
            'figure_path_micro': str(fig_path_micro),
            **export_paths_micro
        }

    # Generar visualización MACRO
    fig_path_macro = Path(Config.FIGURES_DIR) / f'confusion_matrix_macro_{Config.MODEL_TYPE}.png'

    plot_confusion_matrix_macro(
        cm_results_macro=cm_results_macro,
        model_name=model_name + " (Macro)",
        save_path=str(fig_path_macro),
        dpi=300
    )

    # Exportar resultados MACRO
    macro_json_path = Path(Config.STATS_DIR) / f'{model_name.lower().replace(" ", "_")}_macro_confusion_matrix.json'
    macro_json_data = {
        'model_name': model_name + " Macro",
        'validation_tiles': cm_results_macro['n_tiles'],
        'aggregation_method': 'tile-level classification',
        'iou_threshold': cm_results_macro['iou_threshold'],
        'confusion_matrix_tiles': {
            'true_negative': cm_results_macro['tn_tiles'],
            'false_positive': cm_results_macro['fp_tiles'],
            'false_negative': cm_results_macro['fn_tiles'],
            'true_positive': cm_results_macro['tp_tiles']
        },
        'metrics_tile_classification': {
            'accuracy': cm_results_macro['accuracy_tiles'],
            'precision': cm_results_macro['precision_tiles'],
            'recall': cm_results_macro['recall_tiles'],
            'specificity': cm_results_macro['specificity_tiles'],
            'f1_score': cm_results_macro['f1_tiles']
        },
        'iou_distribution': {
            'iou_mean': cm_results_macro['iou_macro'],
            'iou_median': cm_results_macro['iou_median'],
            'iou_std': cm_results_macro['iou_std'],
            'iou_min': cm_results_macro['iou_min'],
            'iou_max': cm_results_macro['iou_max']
        }
    }

    with open(macro_json_path, 'w') as f:
        json.dump(macro_json_data, f, indent=2)

    # Cerrar figura para liberar memoria
    plt.close('all')

    print("\n✓ Análisis MACRO completado")

    # ========================================================================
    # RESUMEN COMPARATIVO
    # ========================================================================
    print("\n" + "="*80)
    print("ANÁLISIS DE MATRIZ DE CONFUSIÓN COMPLETADO (MICRO + MACRO)")
    print("="*80)
    print("\n📊 COMPARACIÓN DE ESCALAS:")
    print("-"*80)
    print(f"{'Métrica':<20} {'MICRO (píxeles)':<20} {'MACRO (teselas)':<20}")
    print("-"*80)
    print(f"{'IoU (promedio)':<20} {cm_results_micro['iou']:>18.4f}  {cm_results_macro['iou_macro']:>18.4f}")
    print(f"{'F1-Score':<20} {cm_results_micro['f1']:>18.4f}  {cm_results_macro['f1_tiles']:>18.4f}")
    print(f"{'Accuracy':<20} {cm_results_micro['accuracy']:>18.4f}  {cm_results_macro['accuracy_tiles']:>18.4f}")
    print(f"{'Precision':<20} {cm_results_micro['precision']:>18.4f}  {cm_results_macro['precision_tiles']:>18.4f}")
    print(f"{'Recall':<20} {cm_results_micro['recall']:>18.4f}  {cm_results_macro['recall_tiles']:>18.4f}")
    print("-"*80)
    print("\n📊 CLASIFICACIÓN DE TESELAS (MACRO):")
    print(f"   TP: {cm_results_macro['tp_tiles']} teselas (bien clasificadas, IoU ≥ {cm_results_macro['iou_threshold']:.2f})")
    print(f"   FN: {cm_results_macro['fn_tiles']} teselas (mal clasificadas, IoU < {cm_results_macro['iou_threshold']:.2f})")
    print(f"   TN: {cm_results_macro['tn_tiles']} teselas (sin manglar, correctas)")
    print(f"   FP: {cm_results_macro['fp_tiles']} teselas (sin manglar, con FP significativos)")
    print("\n💡 INTERPRETACIÓN:")
    print("   - MICRO: Rendimiento operativo a nivel de píxel (agregación global)")
    print("   - IoU MACRO: Rendimiento en teselas desafiantes (manglar disperso, bordes)")
    print("   - Diferencia: Revela impacto de heterogeneidad espacial en el modelo")
    print("\n📈 FIGURAS GENERADAS:")
    print(f"   MICRO: {fig_path_micro}")
    print(f"   MACRO: {fig_path_macro}")
    print("\n📋 MÉTRICAS EXPORTADAS:")
    print(f"   MICRO JSON: {export_paths_micro['json_path']}")
    print(f"   MICRO TXT:  {export_paths_micro['txt_path']}")
    print(f"   MACRO JSON: {macro_json_path}")
    print("="*80 + "\n")

    return {
        'cm_results_micro': cm_results_micro,
        'cm_results_macro': cm_results_macro,
        'figure_path_micro': str(fig_path_micro),
        'figure_path_macro': str(fig_path_macro),
        **export_paths_micro,
        'macro_json_path': str(macro_json_path)
    }


def export_macro_vs_micro_comparison_table(
    metrics_callback: 'MetricsHistoryCallback',
    cm_analysis: dict,
    output_dir: str = None
) -> str:
    """
    Genera una tabla comparativa detallada entre métricas macro y micro.

    Esta función crea un archivo de texto con formato de tabla que compara:
    - Métricas macro del training loop (por tesela)
    - Métricas micro de la matriz de confusión (agregación de píxeles)
    - Diferencias absolutas y porcentuales

    Args:
        metrics_callback: Callback con métricas del entrenamiento
        cm_analysis: Dict con resultados de confusion matrix analysis
        output_dir: Directorio de salida (default: Config.STATS_DIR)

    Returns:
        Ruta del archivo generado
    """
    if output_dir is None:
        output_dir = Config.STATS_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Obtener métricas micro y macro de confusion matrix
    cm_micro = cm_analysis['cm_results_micro']
    cm_macro = cm_analysis['cm_results_macro']

    # Crear archivo de comparación
    model_name = Config.RUN_NAME.lower().replace(' ', '_').replace('-', '_')
    output_path = Path(output_dir) / f'{model_name}_macro_vs_micro_comparison.txt'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"COMPARACIÓN DE MÉTRICAS: MACRO-AVERAGED vs MICRO-AVERAGED\n")
        f.write(f"Modelo: {Config.RUN_NAME}\n")
        f.write(f"Mejor época: {metrics_callback.best_epoch}\n")
        f.write("=" * 100 + "\n\n")

        f.write("CONCEPTOS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"• MACRO (clasificación por tesela - IoU umbral ≥ {cm_macro['iou_threshold']:.2f}):\n")
        f.write("  - Clasifica CADA TESELA como unidad completa (bien/mal clasificada)\n")
        f.write("  - Tesela con manglar: TP si IoU ≥ umbral, FN si IoU < umbral\n")
        f.write("  - Cada tesela tiene el MISMO PESO independientemente de tamaño/contenido\n")
        f.write("  - Revela cuántas teselas son problemáticas (desafiantes espacialmente)\n")
        f.write("  - Útil para identificar regiones geográficas específicas con mal rendimiento\n\n")

        f.write("• MICRO (agregación de píxeles):\n")
        f.write("  - AGREGA todos los píxeles de todas las teselas globalmente\n")
        f.write("  - Luego calcula métrica sobre la agregación global\n")
        f.write("  - Teselas grandes o con mucho manglar tienen MÁS PESO\n")
        f.write("  - Refleja rendimiento operativo a nivel de píxel\n")
        f.write("  - Estándar en literatura de segmentación semántica\n")
        f.write("-" * 100 + "\n\n")

        f.write("TABLA COMPARATIVA DE MÉTRICAS:\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Métrica':<20} {'MICRO (píxel)':<20} {'MACRO (tesela)':<20} {'Diferencia (M-Ma)':<20} {'Δ %':<15}\n")
        f.write("=" * 100 + "\n")

        # IoU (usar IoU promedio del training loop para comparar con macro)
        iou_macro_avg = cm_macro['iou_macro']
        iou_diff = cm_micro['iou'] - iou_macro_avg
        iou_diff_pct = (iou_diff / iou_macro_avg) * 100
        f.write(f"{'IoU (promedio)':<20} {cm_micro['iou']:>18.4f}  {iou_macro_avg:>18.4f}  {iou_diff:>+18.4f}  {iou_diff_pct:>+13.2f}%\n")

        # F1-Score (clasificación)
        f1_macro_tiles = cm_macro['f1_tiles']
        f1_diff = cm_micro['f1'] - f1_macro_tiles
        f1_diff_pct = (f1_diff / f1_macro_tiles) * 100 if f1_macro_tiles > 0 else 0
        f.write(f"{'F1-Score':<20} {cm_micro['f1']:>18.4f}  {f1_macro_tiles:>18.4f}  {f1_diff:>+18.4f}  {f1_diff_pct:>+13.2f}%\n")

        # Accuracy
        acc_macro_tiles = cm_macro['accuracy_tiles']
        acc_diff = cm_micro['accuracy'] - acc_macro_tiles
        acc_diff_pct = (acc_diff / acc_macro_tiles) * 100
        f.write(f"{'Accuracy':<20} {cm_micro['accuracy']:>18.4f}  {acc_macro_tiles:>18.4f}  {acc_diff:>+18.4f}  {acc_diff_pct:>+13.2f}%\n")

        # Precision
        pre_macro_tiles = cm_macro['precision_tiles']
        pre_diff = cm_micro['precision'] - pre_macro_tiles
        pre_diff_pct = (pre_diff / pre_macro_tiles) * 100 if pre_macro_tiles > 0 else 0
        f.write(f"{'Precision':<20} {cm_micro['precision']:>18.4f}  {pre_macro_tiles:>18.4f}  {pre_diff:>+18.4f}  {pre_diff_pct:>+13.2f}%\n")

        # Recall
        rec_macro_tiles = cm_macro['recall_tiles']
        rec_diff = cm_micro['recall'] - rec_macro_tiles
        rec_diff_pct = (rec_diff / rec_macro_tiles) * 100 if rec_macro_tiles > 0 else 0
        f.write(f"{'Recall':<20} {cm_micro['recall']:>18.4f}  {rec_macro_tiles:>18.4f}  {rec_diff:>+18.4f}  {rec_diff_pct:>+13.2f}%\n")

        f.write("=" * 100 + "\n\n")

        # Clasificación de teselas
        f.write("CLASIFICACIÓN DE TESELAS (MACRO):\n")
        f.write("-" * 100 + "\n")
        f.write(f"  Total de teselas: {cm_macro['n_tiles']}\n")
        f.write(f"  TP (bien clasificadas): {cm_macro['tp_tiles']} teselas\n")
        f.write(f"  FN (mal clasificadas):  {cm_macro['fn_tiles']} teselas\n")
        f.write(f"  TN (sin manglar, OK):   {cm_macro['tn_tiles']} teselas\n")
        f.write(f"  FP (sin manglar, FP):   {cm_macro['fp_tiles']} teselas\n")
        f.write(f"  Umbral de clasificación: IoU ≥ {cm_macro['iou_threshold']:.2f}\n")
        f.write("=" * 100 + "\n\n")

        # Estadísticas de distribución (solo macro)
        f.write("ESTADÍSTICAS DE DISTRIBUCIÓN (MACRO - por tesela):\n")
        f.write("-" * 100 + "\n")
        f.write(f"  Teselas evaluadas: {cm_macro['n_tiles']}\n")
        f.write(f"  IoU medio:         {cm_macro['iou_macro']:.4f}\n")
        f.write(f"  IoU mediana:       {cm_macro['iou_median']:.4f}\n")
        f.write(f"  IoU std dev:       {cm_macro['iou_std']:.4f}\n")
        f.write(f"  IoU mínimo:        {cm_macro['iou_min']:.4f}\n")
        f.write(f"  IoU máximo:        {cm_macro['iou_max']:.4f}\n")
        f.write("-" * 100 + "\n\n")

        # Interpretación
        f.write("INTERPRETACIÓN:\n")
        f.write("-" * 100 + "\n")
        iou_macro_avg = cm_macro['iou_macro']
        iou_diff_interp = cm_micro['iou'] - iou_macro_avg
        if iou_diff_interp > 0:
            f.write(f"✓ IoU micro ({cm_micro['iou']:.4f}) > IoU macro ({iou_macro_avg:.4f})\n")
            f.write(f"  → El modelo tiene MEJOR rendimiento a nivel de píxel que a nivel de tesela\n")
            f.write(f"  → Posible sesgo hacia teselas grandes o con manglar continuo\n")
            f.write(f"  → Áreas pequeñas o dispersas pueden estar mal clasificadas\n")
        else:
            f.write(f"✓ IoU macro ({iou_macro_avg:.4f}) > IoU micro ({cm_micro['iou']:.4f})\n")
            f.write(f"  → El modelo es más robusto a nivel de tesela\n")
            f.write(f"  → Buen rendimiento en teselas desafiantes\n")

        iou_diff_pct_interp = (iou_diff_interp / iou_macro_avg) * 100
        f.write(f"\n📊 Diferencia absoluta: {abs(iou_diff_interp):.4f} ({abs(iou_diff_pct_interp):.2f}%)\n")
        if abs(iou_diff_interp) < 0.02:
            f.write("  → Diferencia BAJA: Rendimiento consistente entre escalas\n")
        elif abs(iou_diff_interp) < 0.05:
            f.write("  → Diferencia MODERADA: Revisar casos desafiantes\n")
        else:
            f.write("  → Diferencia ALTA: Heterogeneidad espacial significativa\n")

        f.write("-" * 100 + "\n\n")

        # Recomendaciones para reporte
        f.write("RECOMENDACIONES PARA PUBLICACIÓN:\n")
        f.write("-" * 100 + "\n")
        f.write("1. REPORTAR AMBAS MÉTRICAS:\n")
        f.write("   - Métricas micro en tabla principal (estándar de comparación con literatura)\n")
        f.write("   - Métricas macro en tabla suplementaria (análisis de robustez espacial)\n\n")
        f.write("2. JUSTIFICACIÓN METODOLÓGICA:\n")
        f.write("   'Se reportan métricas micro-averaged (agregación global de píxeles) para\n")
        f.write("    comparación con literatura, y macro-averaged (promedio por tesela) para\n")
        f.write("    evaluar robustez en regiones espacialmente heterogéneas.'\n\n")
        f.write("3. INTERPRETACIÓN DE DIFERENCIAS:\n")
        f.write("   'La diferencia entre IoU micro y macro refleja el impacto de la\n")
        f.write("    heterogeneidad espacial en el rendimiento del modelo.'\n")
        f.write("-" * 100 + "\n\n")

        f.write("=" * 100 + "\n")
        f.write(f"Archivo generado: {output_path}\n")
        f.write(f"Fecha: {Path(output_path).stat().st_mtime}\n")
        f.write("=" * 100 + "\n")

    print(f"✓ Tabla comparativa guardada en: {output_path}")
    return str(output_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal de entrenamiento."""

    # Imprimir configuración
    print_config()

    # ========================================================================
    # PASO 1: CONFIGURACIÓN DE DIRECTORIOS
    # ========================================================================
    setup_output_directories()

    # ========================================================================
    # PASO 2: CREAR DATAMODULE Y ANÁLISIS DE DISTRIBUCIÓN
    # ========================================================================
    print("🔧 Creando componentes...")
    dm = create_datamodule()

    # Análisis de distribución de píxeles (antes del entrenamiento)
    compute_pixel_statistics(dm)

    # ========================================================================
    # PASO 3: BRANCHING - RANDOM FOREST vs CNNs
    # ========================================================================
    # Random Forest usa pipeline completamente diferente (no PyTorch Lightning)

    if Config.MODEL_TYPE == "random_forest":
        # ====================================================================
        # PIPELINE RANDOM FOREST (ML CLÁSICO)
        # ====================================================================
        print("\n" + "="*80)
        print("🌲 EJECUTANDO PIPELINE RANDOM FOREST")
        print("="*80 + "\n")

        # Validar configuración
        model = create_model()  # Solo para validar config, retorna None

        # Ejecutar pipeline completo de RF
        rf_model, rf_metrics = train_random_forest_pipeline(dm)

        print("\n" + "="*80)
        print("ENTRENAMIENTO RANDOM FOREST COMPLETADO")
        print("="*80)

        # Resumen final
        print("\n" + "="*80)
        print("RESUMEN DE OUTPUTS GENERADOS - RANDOM FOREST")
        print("="*80)
        print(f"📊 Estadísticas (Train): {Config.STATS_DIR}/pixel_counts_train.txt")
        print(f"📊 Estadísticas (Val):   {Config.STATS_DIR}/pixel_counts_val.txt")
        print(f"📈 Distribución comparativa: {Config.FIGURES_DIR}/class_distribution.jpg")
        print(f"💾 Modelo Random Forest: {Config.CHECKPOINT_DIR}/random_forest_model.pkl")
        print(f"📋 Métricas (TXT): {Config.STATS_DIR}/random_forest_metrics.txt")
        print(f"📋 Métricas (JSON): {Config.STATS_DIR}/random_forest_metrics.json")
        print("\n📊 MÉTRICAS FINALES:")
        print(f"   IoU:       {rf_metrics['iou']:.4f}")
        print(f"   F1-Score:  {rf_metrics['f1']:.4f}")
        print(f"   Kappa:     {rf_metrics['kappa']:.4f}")
        print(f"   Accuracy:  {rf_metrics['oa']:.4f}")
        print(f"   Precision: {rf_metrics['precision']:.4f}")
        print(f"   Recall:    {rf_metrics['recall']:.4f}")
        print("\n💡 COMPARACIÓN CON CNNs:")
        print("   Para comparar con modelos CNN, ejecuta el script con otros MODEL_TYPE")
        print("   y compara las métricas en outputs/stats/")
        print("="*80 + "\n")

    else:
        # ====================================================================
        # PIPELINE CNN (PYTORCH LIGHTNING)
        # ====================================================================
        # ========================================================================
        # PASO 3: CREAR MODELO Y COMPONENTES DE ENTRENAMIENTO
        # ========================================================================
        model = create_model()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(optimizer)
        loss_fn = create_loss_fn()
        metrics = create_metrics()

        # Crear LightningModule
        module = create_lightning_module(model, optimizer, scheduler, loss_fn, metrics)

        # ========================================================================
        # PASO 4: CONFIGURAR CALLBACKS Y TRAINER
        # ========================================================================
        callbacks = create_callbacks()
        trainer = create_trainer(callbacks)

        # ========================================================================
        # PASO 5: ENTRENAMIENTO
        # ========================================================================
        print("\n" + "="*80)
        print("INICIANDO ENTRENAMIENTO")
        print("="*80 + "\n")

        trainer.fit(module, dm)

        print("\n" + "="*80)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*80)

        # ========================================================================
        # PASO 6: ANÁLISIS POST-ENTRENAMIENTO
        # ========================================================================
        # La figura de convergencia (loss.jpg) se genera automáticamente
        # por el LossHistoryCallback al finalizar el entrenamiento

        # Analizar checkpoints
        analyze_checkpoints()

        # ================================================================
        # ANÁLISIS DE MATRIZ DE CONFUSIÓN
        # ================================================================
        # Calcular y visualizar matriz de confusión sobre conjunto de validación
        cm_analysis = run_confusion_matrix_analysis(dm=dm, model_fn=create_model)

        # ================================================================
        # OBTENER CALLBACK DE MÉTRICAS
        # ================================================================
        # Extraer el callback que contiene las métricas del entrenamiento
        metrics_callback = None
        for callback in callbacks:
            if isinstance(callback, MetricsHistoryCallback):
                metrics_callback = callback
                break

        # ================================================================
        # TABLA COMPARATIVA MACRO VS MICRO
        # ================================================================
        # Generar tabla detallada comparando ambas escalas de agregación
        if cm_analysis and metrics_callback:
            print("\n" + "="*80)
            print("GENERANDO TABLA COMPARATIVA MACRO VS MICRO")
            print("="*80)
            comparison_table_path = export_macro_vs_micro_comparison_table(
                metrics_callback=metrics_callback,
                cm_analysis=cm_analysis,
                output_dir=Config.STATS_DIR
            )
            print("="*80 + "\n")

        # ================================================================
        # GUARDAR MÉTRICAS EN JSON (para comparación con compare_all.py)
        # ================================================================
        print("\n" + "="*80)
        print("EXPORTANDO MÉTRICAS A JSON")
        print("="*80)

        if metrics_callback and metrics_callback.val_metrics['iou']:
            # Extraer métricas del mejor modelo
            best_idx = metrics_callback.best_epoch - 1

            # ================================================================
            # MÉTRICAS MACRO-AVERAGED (del loop de entrenamiento/validación)
            # ================================================================
            # Las métricas del training loop son MACRO porque se calcula IoU
            # por tesela dentro de cada batch, luego se promedian
            metrics_dict = {
                # --- MÉTRICAS MACRO (por tesela, del training loop) ---
                'iou_macro': float(metrics_callback.val_metrics['iou'][best_idx]),
                'f1_macro': float(metrics_callback.val_metrics['f1'][best_idx]),
                'kappa_macro': float(metrics_callback.val_metrics['kp'][best_idx]),
                'accuracy_macro': float(metrics_callback.val_metrics['OA'][best_idx]),
                'precision_macro': float(metrics_callback.val_metrics['pre'][best_idx]),
                'recall_macro': float(metrics_callback.val_metrics['recall'][best_idx]),
                'fwiou_macro': float(metrics_callback.val_metrics['FWIoU'][best_idx]),

                # Información del modelo
                'model_type': Config.MODEL_TYPE,
                'model_name': Config.RUN_NAME,
                'best_epoch': int(metrics_callback.best_epoch)
            }

            # ================================================================
            # MÉTRICAS MICRO-AVERAGED (de confusion matrix analysis)
            # ================================================================
            # Extraer métricas micro de la matriz de confusión si están disponibles
            if cm_analysis and 'cm_results_micro' in cm_analysis:
                cm_micro = cm_analysis['cm_results_micro']
                metrics_dict.update({
                    'iou_micro': float(cm_micro['iou']),
                    'f1_micro': float(cm_micro['f1']),
                    'accuracy_micro': float(cm_micro['accuracy']),
                    'precision_micro': float(cm_micro['precision']),
                    'recall_micro': float(cm_micro['recall']),
                    'specificity_micro': float(cm_micro['specificity'])
                })

            # ================================================================
            # MÉTRICAS MACRO ADICIONALES (de confusion matrix analysis)
            # ================================================================
            # Incluir estadísticas de distribución de IoU por tesela y clasificación de teselas
            if cm_analysis and 'cm_results_macro' in cm_analysis:
                cm_macro = cm_analysis['cm_results_macro']
                metrics_dict.update({
                    # Distribución de IoU
                    'iou_macro_std': float(cm_macro['iou_std']),
                    'iou_macro_min': float(cm_macro['iou_min']),
                    'iou_macro_max': float(cm_macro['iou_max']),
                    'iou_macro_median': float(cm_macro['iou_median']),
                    # Clasificación de teselas (usando umbral de IoU)
                    'f1_tiles': float(cm_macro['f1_tiles']),
                    'accuracy_tiles': float(cm_macro['accuracy_tiles']),
                    'precision_tiles': float(cm_macro['precision_tiles']),
                    'recall_tiles': float(cm_macro['recall_tiles']),
                    # Conteo de teselas
                    'tp_tiles': int(cm_macro['tp_tiles']),
                    'fn_tiles': int(cm_macro['fn_tiles']),
                    'tn_tiles': int(cm_macro['tn_tiles']),
                    'fp_tiles': int(cm_macro['fp_tiles']),
                    'iou_threshold': float(cm_macro['iou_threshold'])
                })

            # ================================================================
            # MÉTRICAS LEGACY (para compatibilidad hacia atrás)
            # ================================================================
            # Mantener nombres antiguos apuntando a micro (estándar en literatura)
            if cm_analysis and 'cm_results_micro' in cm_analysis:
                cm_micro = cm_analysis['cm_results_micro']
                metrics_dict.update({
                    'iou': float(cm_micro['iou']),  # Por defecto = micro
                    'f1': float(cm_micro['f1']),
                    'accuracy': float(cm_micro['accuracy']),
                    'precision': float(cm_micro['precision']),
                    'recall': float(cm_micro['recall'])
                })
            else:
                # Fallback: usar macro si no hay micro disponible
                metrics_dict.update({
                    'iou': float(metrics_callback.val_metrics['iou'][best_idx]),
                    'f1': float(metrics_callback.val_metrics['f1'][best_idx]),
                    'accuracy': float(metrics_callback.val_metrics['OA'][best_idx]),
                    'precision': float(metrics_callback.val_metrics['pre'][best_idx]),
                    'recall': float(metrics_callback.val_metrics['recall'][best_idx])
                })

            # Agregar nota metodológica
            metrics_dict['_note'] = (
                "Métricas *_macro: calculadas por tesela y promediadas (cada tesela tiene el mismo peso). "
                "Métricas *_micro: agregación global de píxeles antes de calcular (standard en segmentación semántica). "
                "Ver README para diferencias metodológicas."
            )

            # Generar nombre de archivo basado en configuración
            model_name = Config.RUN_NAME.lower().replace('-', '_').replace(' ', '_')
            json_file = Path(Config.STATS_DIR) / f'{model_name}_metrics.json'

            # Asegurar que el directorio existe
            os.makedirs(Config.STATS_DIR, exist_ok=True)

            # Guardar JSON
            with open(json_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)

            print(f"✓ Métricas JSON guardadas en: {json_file}")
            print(f"\n📊 MÉTRICAS DEL MEJOR MODELO (época {metrics_callback.best_epoch}):")
            print("="*80)

            # Mostrar métricas macro (del training loop)
            print("\n🎯 MÉTRICAS MACRO-AVERAGED (por tesela):")
            print(f"   IoU:       {metrics_dict['iou_macro']:.4f}")
            print(f"   F1-Score:  {metrics_dict['f1_macro']:.4f}")
            print(f"   Precision: {metrics_dict['precision_macro']:.4f}")
            print(f"   Recall:    {metrics_dict['recall_macro']:.4f}")
            print(f"   Accuracy:  {metrics_dict['accuracy_macro']:.4f}")
            print(f"   Kappa:     {metrics_dict['kappa_macro']:.4f}")

            # Mostrar métricas micro si están disponibles
            if 'iou_micro' in metrics_dict:
                print("\n🔬 MÉTRICAS MICRO-AVERAGED (agregación de píxeles):")
                print(f"   IoU:       {metrics_dict['iou_micro']:.4f}")
                print(f"   F1-Score:  {metrics_dict['f1_micro']:.4f}")
                print(f"   Precision: {metrics_dict['precision_micro']:.4f}")
                print(f"   Recall:    {metrics_dict['recall_micro']:.4f}")
                print(f"   Accuracy:  {metrics_dict['accuracy_micro']:.4f}")

                # Mostrar diferencia entre macro y micro
                iou_diff = metrics_dict['iou_micro'] - metrics_dict['iou_macro']
                print(f"\n📊 DIFERENCIA (Micro - Macro):")
                print(f"   ΔIoU:      {iou_diff:+.4f} ({iou_diff*100:+.2f}%)")

            print("="*80 + "\n")
        else:
            print("⚠️  No se pudieron extraer métricas del callback")
            print("="*80 + "\n")

        # Resumen final
        print("\n" + "="*80)
        print("RESUMEN DE OUTPUTS GENERADOS")
        print("="*80)
        print(f"📊 Estadísticas (Train): {Config.STATS_DIR}/pixel_counts_train.txt")
        print(f"📊 Estadísticas (Val):   {Config.STATS_DIR}/pixel_counts_val.txt")
        print(f"📈 Distribución comparativa: {Config.FIGURES_DIR}/class_distribution.jpg")
        print(f"📉 Convergencia: {Config.FIGURES_DIR}/loss.jpg")
        print(f"📋 Métricas de entrenamiento: {Config.STATS_DIR}/training_metrics_*.txt")
        if metrics_callback and metrics_callback.val_metrics['iou']:
            print(f"📋 Métricas JSON: {Config.STATS_DIR}/{model_name}_metrics.json")
        print(f"📊 Matriz de confusión MICRO: {Config.FIGURES_DIR}/confusion_matrix_micro_{Config.MODEL_TYPE}.png")
        print(f"📊 Matriz de confusión MACRO: {Config.FIGURES_DIR}/confusion_matrix_macro_{Config.MODEL_TYPE}.png")
        if cm_analysis:
            print(f"📋 Matriz confusión MICRO JSON: {cm_analysis.get('json_path', 'N/A')}")
            print(f"📋 Matriz confusión MACRO JSON: {cm_analysis.get('macro_json_path', 'N/A')}")
        if cm_analysis and metrics_callback:
            print(f"📊 Tabla comparativa MACRO vs MICRO: {Config.STATS_DIR}/{model_name}_macro_vs_micro_comparison.txt")
        print(f"💾 Checkpoints: {Config.CHECKPOINT_DIR}/")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
