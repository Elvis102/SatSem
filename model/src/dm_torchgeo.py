"""
DataModule de PyTorch Lightning para trabajar con TorchGeo.

Este DataModule organiza los datos de entrenamiento, validación y prueba
usando el dataset de TorchGeo (SatSegDataset) y soporta transformaciones
específicas de TorchGeo que funcionan mejor con datos multiespectrales.

================================================================================
CONFIGURACIÓN DE DIRECTORIOS DE DATOS
================================================================================

Este DataModule soporta DOS configuraciones de directorios:

CONFIGURACIÓN A (ORIGINAL - 220 teselas training):
    train_dir = 'train'
    - Busca imágenes en: {data_dir}/train/Manglar_2021_images/*.tif
    - Busca máscaras en: {data_dir}/train/Manglar_2021_masks/*.tif
    - Patrón: 2021_Sentinel-2_r000_c022.tif -> 2021_Sentinel-2_mask_r000_c022.tif
    - Total: 220 teselas de entrenamiento
    - Resultado esperado: Val IoU ~0.9486

CONFIGURACIÓN B (ALTERNATIVA - 176 teselas training):
    train_dir = 'train/train'
    - Busca imágenes en: {data_dir}/train/train/tile_*.tif
    - Busca máscaras en: {data_dir}/train/train/mask_*.tif
    - Patrón: tile_0000.tif -> mask_0000.tif
    - Total: 176 teselas de entrenamiento
    - Resultado esperado: Val IoU ~0.90

Para cambiar entre configuraciones, modificar el parámetro `train_dir` en __init__
o pasarlo explícitamente al crear el DataModule.

================================================================================
"""

import lightning as L
from typing import Callable, Optional, Sequence
from torch.utils.data import DataLoader
from pathlib import Path

from .ds_torchgeo import SatSegDataset


class TorchGeoDataModule(L.LightningDataModule):
    """
    DataModule de Lightning para trabajar con TorchGeo.

    Este DataModule organiza los datos de entrenamiento y validación usando el
    dataset de TorchGeo (SatSegDataset) y soporta transformaciones específicas
    de TorchGeo que funcionan mejor con datos multiespectrales.

    Args:
        data_dir: Directorio raíz donde se encuentran los datos
        train_dir: Subdirectorio con datos de entrenamiento (relativo a data_dir).
                   - 'train' (default): usa Manglar_2021_images/ (220 teselas)
                   - 'train/train': usa tile_*.tif (176 teselas)
        val_dir: Subdirectorio con datos de validación (relativo a data_dir).
                 Debe contener archivos tile_XXXX.tif y mask_XXXX.tif
        test_dir: (Opcional) Subdirectorio con datos de prueba
        batch_size: Tamaño del batch para el entrenamiento
        num_workers: Número de procesos paralelos para cargar datos
        pin_memory: Si True, usa memoria pinned para transferencia GPU más rápida
        train_trans: Transformaciones de TorchGeo para entrenamiento (AugmentationSequential)
        val_trans: Transformaciones de TorchGeo para validación
        test_trans: Transformaciones de TorchGeo para prueba
        cache_data: Si True, cachea los datos en memoria RAM
        bands: Índices de las bandas espectrales a usar
        norm_value: Factor de normalización para las imágenes
        verbose: Si True, imprime información de configuración

    Example:
        >>> # Configuración A (220 teselas - default)
        >>> dm = TorchGeoDataModule(
        ...     data_dir='/path/to/data',
        ...     train_dir='train',  # Usa Manglar_2021_images/
        ...     batch_size=32,
        ... )
        >>>
        >>> # Configuración B (176 teselas)
        >>> dm = TorchGeoDataModule(
        ...     data_dir='/path/to/data',
        ...     train_dir='train/train',  # Usa tile_*.tif
        ...     batch_size=32,
        ... )
    """

    def __init__(
        self,
        data_dir: str,
        train_dir: str = 'train/train',  # CONFIGURACIÓN B: train/train/ (176 teselas, SIN data leakage)
        val_dir: str = 'train/val',
        test_dir: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_trans: Optional[Callable] = None,
        val_trans: Optional[Callable] = None,
        test_trans: Optional[Callable] = None,
        cache_data: bool = False,
        bands: Optional[Sequence[int]] = (3, 2, 1, 7),
        norm_value: float = 3.0,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans

        self.cache_data = cache_data
        self.bands = bands
        self.norm_value = norm_value
        self.verbose = verbose

        # Calcular número de bandas
        self.num_bands = len(bands) if bands is not None else 12

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Configura los datasets de entrenamiento, validación y prueba.

        Args:
            stage: 'fit', 'validate', 'test', o None (configura todos)
        """

        # ====================================================================
        # TRAINING DATASET
        # ====================================================================
        if stage == "fit" or stage is None:
            train_root = self.data_dir / self.train_dir

            # Detectar automáticamente el formato de datos
            # CONFIGURACIÓN A: Manglar_2021_images/ con patrón 2021_Sentinel-2_*.tif
            train_images_dir = train_root / 'Manglar_2021_images'
            train_masks_dir = train_root / 'Manglar_2021_masks'

            if train_images_dir.exists():
                # ============================================================
                # CONFIGURACIÓN A: Manglar_2021_images/ (220 teselas)
                # ============================================================
                all_images = sorted(list(train_images_dir.glob('*.tif')))

                if len(all_images) == 0:
                    raise FileNotFoundError(
                        f"No training images found in: {train_images_dir}\n"
                        f"Please verify the path and pattern (*.tif)"
                    )

                # Crear lista de rutas
                train_tiles = [str(p) for p in all_images]

                # Generar rutas de máscaras
                # 2021_Sentinel-2_r000_c022.tif -> 2021_Sentinel-2_mask_r000_c022.tif
                train_masks = [
                    str(train_masks_dir / p.name.replace('2021_Sentinel-2_', '2021_Sentinel-2_mask_'))
                    for p in all_images
                ]

                if self.verbose:
                    print(f"📁 Usando CONFIGURACIÓN A: {train_images_dir}")
            else:
                # ============================================================
                # CONFIGURACIÓN B: train/train/ con patrón tile_*.tif (176 teselas)
                # ============================================================
                all_images = sorted(list(train_root.glob('tile_*.tif')))

                if len(all_images) == 0:
                    raise FileNotFoundError(
                        f"No training images found in: {train_root}\n"
                        f"Please verify the path and pattern (tile_*.tif)\n"
                        f"Or check if Manglar_2021_images/ exists"
                    )

                # Crear lista de rutas
                train_tiles = [str(p) for p in all_images]

                # Generar rutas de máscaras: tile_0000.tif -> mask_0000.tif
                train_masks = [
                    str(train_root / p.name.replace('tile_', 'mask_'))
                    for p in all_images
                ]

                if self.verbose:
                    print(f"📁 Usando CONFIGURACIÓN B: {train_root}")

            # Crear dataset de entrenamiento
            self.train_ds = SatSegDataset(
                images=train_tiles,
                masks=train_masks,
                bands=self.bands,
                norm_value=self.norm_value,
                transforms=self.train_trans,
                cache_data=self.cache_data,
            )

            if self.verbose:
                print(f"✓ Training samples: {len(train_tiles)}")

        # ====================================================================
        # VALIDATION DATASET
        # ====================================================================
        if stage == "fit" or stage == "validate" or stage is None:
            val_root = self.data_dir / self.val_dir

            # En validación: tile_0000.tif y mask_0000.tif en el mismo directorio
            all_val_images = sorted(list(val_root.glob('tile_*.tif')))

            if len(all_val_images) == 0:
                raise FileNotFoundError(
                    f"No validation images found in: {val_root}\n"
                    f"Please verify the path and pattern (tile_*.tif)"
                )

            val_tiles = [str(p) for p in all_val_images]

            # Generar rutas de máscaras: tile_0000.tif -> mask_0000.tif
            val_masks = [
                str(val_root / p.name.replace('tile_', 'mask_'))
                for p in all_val_images
            ]

            # Crear dataset de validación
            self.val_ds = SatSegDataset(
                images=val_tiles,
                masks=val_masks,
                bands=self.bands,
                norm_value=self.norm_value,
                transforms=self.val_trans,
                cache_data=self.cache_data,
            )

            if self.verbose:
                print(f"✓ Validation samples: {len(val_tiles)}")

        # ====================================================================
        # TEST DATASET (opcional)
        # ====================================================================
        if stage == "test" and self.test_dir is not None:
            test_root = self.data_dir / self.test_dir

            # Usar el mismo patrón que validación
            all_test_images = sorted(list(test_root.glob('tile_*.tif')))

            if len(all_test_images) == 0:
                raise FileNotFoundError(
                    f"No test images found in: {test_root}\n"
                    f"Please verify the path and pattern (tile_*.tif)"
                )

            test_tiles = [str(p) for p in all_test_images]
            test_masks = [
                str(test_root / p.name.replace('tile_', 'mask_'))
                for p in all_test_images
            ]

            # Crear dataset de prueba
            self.test_ds = SatSegDataset(
                images=test_tiles,
                masks=test_masks,
                bands=self.bands,
                norm_value=self.norm_value,
                transforms=self.test_trans,
                cache_data=self.cache_data,
            )

            if self.verbose:
                print(f"✓ Test samples: {len(test_tiles)}")

        # Imprimir resumen de configuración
        if self.verbose and stage is None:
            print(f"✓ Bands: {self.bands}")
            print(f"✓ Number of bands: {self.num_bands}")
            print(f"✓ Normalization value: {self.norm_value}")
            print(f"✓ Cache enabled: {self.cache_data}")
            print(f"✓ Batch size: {self.batch_size}")
            print(f"✓ Num workers: {self.num_workers}")

    def train_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Retorna el DataLoader de entrenamiento.

        Args:
            batch_size: (Opcional) Sobrescribe el batch_size del constructor
            shuffle: Si True, mezcla los datos cada época

        Returns:
            DataLoader configurado para entrenamiento
        """
        return DataLoader(
            self.train_ds,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Retorna el DataLoader de validación.

        Args:
            batch_size: (Opcional) Sobrescribe el batch_size del constructor
            shuffle: Si True, mezcla los datos

        Returns:
            DataLoader configurado para validación
        """
        return DataLoader(
            self.val_ds,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False
    ) -> Optional[DataLoader]:
        """
        Retorna el DataLoader de prueba (si existe).

        Args:
            batch_size: (Opcional) Sobrescribe el batch_size del constructor
            shuffle: Si True, mezcla los datos

        Returns:
            DataLoader configurado para prueba, o None si no hay test set
        """
        if not hasattr(self, 'test_ds'):
            return None

        return DataLoader(
            self.test_ds,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )
