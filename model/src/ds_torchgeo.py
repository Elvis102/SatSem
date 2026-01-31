"""
Dataset de TorchGeo para segmentación de imágenes satelitales.

Este módulo extiende NonGeoDataset de TorchGeo para trabajar con tiles de 
imágenes satelitales y sus máscaras de segmentación correspondientes.
"""

import torch
from typing import Callable, Optional, Sequence
from pathlib import Path
import numpy as np
from torchgeo.datasets import NonGeoDataset
import rasterio


class SatSegDataset(NonGeoDataset):
    """
    Dataset para segmentación de imágenes satelitales usando TorchGeo.
    
    Este dataset extiende NonGeoDataset de TorchGeo para trabajar con tiles de 
    imágenes satelitales y sus máscaras de segmentación correspondientes.
    
    Características:
    - Selección flexible de bandas espectrales
    - Normalización configurable
    - Manejo robusto de valores NaN
    - Soporte para transformaciones de TorchGeo
    - Cacheo opcional de datos en memoria
    
    Args:
        images: Lista de rutas a las imágenes (tiles)
        masks: Lista de rutas a las máscaras correspondientes
        bands: Índices de las bandas a usar (por defecto RGB+NIR: 3,2,1,7)
        norm_value: Valor de normalización para multiplicar las imágenes
        transforms: Transformaciones de TorchGeo a aplicar (AugmentationSequential)
        cache_data: Si True, cachea los datos en memoria para mayor velocidad
        validate_files: Si True, valida que todos los archivos existen
    
    Example:
        >>> dataset = SatSegDataset(
        ...     images=['tile1.tif', 'tile2.tif'],
        ...     masks=['mask1.tif', 'mask2.tif'],
        ...     bands=[0, 1, 2, 3],  # RGB + NIR
        ...     transforms=aug_transforms,
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape, sample['mask'].shape)
    """
    
    def __init__(
        self,
        images: list[str],
        masks: list[str],
        bands: Optional[Sequence[int]] = (3, 2, 1, 7),
        norm_value: float = 3.0,
        transforms: Optional[Callable] = None,
        cache_data: bool = False,
        validate_files: bool = True,
    ) -> None:
        # Inicializar la clase padre sin argumentos
        # NonGeoDataset no requiere argumentos en __init__
        super().__init__()
        
        # Convertir rutas a Path objects
        self.images = [Path(img) for img in images]
        self.masks = [Path(mask) for mask in masks]
        
        self.bands = bands
        self.norm_value = norm_value
        self.transforms = transforms
        self.cache_data = cache_data
        
        # Validar que hay el mismo número de imágenes y máscaras
        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Number of images ({len(self.images)}) != "
                f"number of masks ({len(self.masks)})"
            )
        
        # Validar que los archivos existen
        if validate_files:
            self._validate_files()
        
        # Cachear datos en memoria si se solicita
        if cache_data:
            print(f"Caching {len(self.images)} samples in memory...")
            self._cached_data = [self._load_data(idx) for idx in range(len(self.images))]
        else:
            self._cached_data = None
    
    def _validate_files(self) -> None:
        """Valida que todos los archivos de imagen y máscara existen."""
        missing_files = []
        
        for img, mask in zip(self.images, self.masks):
            if not img.exists():
                missing_files.append(f"Image: {img}")
            if not mask.exists():
                missing_files.append(f"Mask: {mask}")
        
        if missing_files:
            raise FileNotFoundError(
                f"The following files are missing:\n" + 
                "\n".join(f"  - {f}" for f in missing_files)
            )
    
    def __len__(self) -> int:
        """Retorna el número total de muestras."""
        return len(self.images)
    
    def __repr__(self) -> str:
        """Representación en string del dataset."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  n_samples={len(self)},\n"
            f"  bands={self.bands},\n"
            f"  num_bands={self.num_bands},\n"
            f"  norm_value={self.norm_value},\n"
            f"  cached={self.cache_data},\n"
            f"  transforms={self.transforms is not None}\n"
            f")"
        )
    
    def _load_data(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Carga una imagen y su máscara desde disco.
        
        Args:
            idx: Índice de la muestra a cargar
            
        Returns:
            Diccionario con 'image' y 'mask' como tensores de torch
        """
        # ====================================================================
        # CARGAR IMAGEN
        # ====================================================================
        with rasterio.open(self.images[idx]) as src:
            image = src.read()  # Shape: (bands, height, width)
        
        # Seleccionar bandas específicas si se especificaron
        if self.bands is not None:
            image = image[list(self.bands), :, :]
        
        # Aplicar normalización
        image = image.astype(np.float32) / self.norm_value
        
        # Reemplazar NaN con 0
        image = np.nan_to_num(image, nan=0.0)
        
        # ====================================================================
        # CARGAR MÁSCARA
        # ====================================================================
        with rasterio.open(self.masks[idx]) as src:
            mask = src.read(1)  # Leer primera banda solamente
        
        # Asegurar que la máscara es binaria y float32
        mask = mask.astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        
        # Agregar dimensión de canal a la máscara
        mask = mask[np.newaxis, :, :]  # Shape: (1, height, width)
        
        # ====================================================================
        # CONVERTIR A TENSORES DE PYTORCH
        # ====================================================================
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        return {'image': image, 'mask': mask}
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retorna una muestra del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Diccionario con 'image' y 'mask' (y potencialmente otras claves 
            si las transformaciones las añaden)
        """
        # Cargar datos (desde caché o disco)
        if self.cache_data and self._cached_data is not None:
            # Crear una copia para evitar modificar el caché
            sample = {
                'image': self._cached_data[idx]['image'].clone(),
                'mask': self._cached_data[idx]['mask'].clone(),
            }
        else:
            sample = self._load_data(idx)
        
        # Aplicar transformaciones de TorchGeo si existen
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
    
    @property
    def num_bands(self) -> int:
        """
        Retorna el número de bandas espectrales.
        
        Returns:
            Número de bandas en las imágenes
        """
        if self.bands is not None:
            return len(self.bands)
        
        # Si no se especificaron bandas, determinar del primer archivo
        with rasterio.open(self.images[0]) as src:
            return src.count
    
    def get_sample_info(self, idx: int) -> dict:
        """
        Obtiene información sobre una muestra sin cargar los datos.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Diccionario con metadatos de la muestra
        """
        with rasterio.open(self.images[idx]) as src:
            info = {
                'image_path': str(self.images[idx]),
                'mask_path': str(self.masks[idx]),
                'width': src.width,
                'height': src.height,
                'num_bands': src.count,
                'dtype': src.dtypes[0],
                'crs': str(src.crs),
                'bounds': src.bounds,
            }
        
        return info
