"""
Satseg-TorchGeo: Pipeline de segmentación de imágenes satelitales con TorchGeo.

Este módulo contiene las clases principales para trabajar con datos geoespaciales
usando TorchGeo, PyTorch Lightning y Segmentation Models PyTorch.
"""

from .ds_torchgeo import SatSegDataset
from .dm_torchgeo import TorchGeoDataModule
from .module import Module
from .metrics import (
    BinarySegmentationMetrics,
    iou,
    f1_score,
    overall_accuracy,
    precision,
    recall,
    fw_iou,
    kappa,
)

__version__ = '0.2.0'

__all__ = [
    # Dataset y DataModule
    'SatSegDataset',
    'TorchGeoDataModule',
    
    # LightningModule
    'Module',
    
    # Clase de métricas
    'BinarySegmentationMetrics',
    
    # Funciones de métricas
    'iou',
    'f1_score',
    'overall_accuracy',
    'precision',
    'recall',
    'fw_iou',
    'kappa',
]
