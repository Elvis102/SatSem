"""
Module base para segmentación con PyTorch Lightning.

Este módulo proporciona una clase base para tareas de segmentación que:
- Es compatible con batches como tuplas (x, y) o diccionarios {'image': x, 'mask': y}
- Maneja schedulers correctamente
- Soporta múltiples métricas
- Proporciona hooks para personalización
"""

import lightning as L
import torch
from typing import Dict, Optional, Callable, Any, Union


class Module(L.LightningModule):
    """
    LightningModule base para segmentación.
    
    Compatible con:
    - Batches como tuplas: (x, y)
    - Batches como diccionarios: {'image': x, 'mask': y}
    - TorchGeo datasets
    - Datasets estándar de PyTorch
    
    Args:
        model: Modelo de segmentación (nn.Module)
        optimizer: Optimizador configurado
        loss_fn: Función de pérdida
        metrics: Diccionario de métricas {nombre: función}
        scheduler: (Opcional) Learning rate scheduler
        scheduler_config: (Opcional) Configuración del scheduler
    
    Example:
        >>> model = smp.Unet(...)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> loss_fn = smp.losses.DiceLoss(mode='binary')
        >>> metrics = {'iou': iou_metric, 'f1': f1_metric}
        >>> 
        >>> module = Module(model, optimizer, loss_fn, metrics)
        >>> trainer = L.Trainer(max_epochs=10)
        >>> trainer.fit(module, datamodule)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        
        # Guardar hiperparámetros (excluir objetos no serializables)
        self.save_hyperparameters(ignore=['model', 'optimizer', 'loss_fn', 'metrics', 'scheduler'])
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.scheduler = scheduler
        
        # Configuración por defecto para scheduler
        self.scheduler_config = scheduler_config or {
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo."""
        return self.model(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predicción con umbralización.
        
        Args:
            x: Imagen de entrada
            threshold: Umbral para binarización
            
        Returns:
            Máscara binaria predicha
        """
        self.eval()
        with torch.no_grad():
            y_pred = self(x.to(self.device))
            return torch.sigmoid(y_pred) > threshold
    
    def _unpack_batch(
        self, 
        batch: Union[tuple, Dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Desempaqueta batch en formato (x, y).
        
        Soporta:
        - Tuplas: (x, y)
        - Diccionarios: {'image': x, 'mask': y}
        
        También maneja dimensiones extra agregadas por algunos dataloaders.
        
        Args:
            batch: Batch del dataloader
            
        Returns:
            Tupla (x, y) con tensores limpios
        """
        # Desempaquetar según tipo
        if isinstance(batch, dict):
            x = batch['image']
            y = batch['mask']
        else:
            x, y = batch
        
        # Eliminar dimensiones extra si existen
        # Caso: x tiene shape [B, 1, C, H, W] -> squeeze a [B, C, H, W]
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        
        # Caso: y tiene shape [B, 1, 1, H, W] -> squeeze a [B, 1, H, W]
        while y.dim() == 5:
            y = y.squeeze(1)
        
        return x, y
    
    def shared_step(
        self, 
        batch: Union[tuple, Dict[str, torch.Tensor]], 
        batch_idx: int
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Paso compartido entre training y validation.
        
        Args:
            batch: Batch de datos
            batch_idx: Índice del batch
            
        Returns:
            Tupla (loss, metrics_dict)
        """
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Calcular métricas
        metrics_dict = {}
        for name, metric_fn in self.metrics.items():
            metrics_dict[name] = metric_fn(y_hat, y)
        
        return loss, metrics_dict
    
    def training_step(
        self, 
        batch: Union[tuple, Dict[str, torch.Tensor]], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch de datos
            batch_idx: Índice del batch
            
        Returns:
            Loss para backpropagation
        """
        loss, metrics = self.shared_step(batch, batch_idx)
        
        # Logear loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Logear métricas
        for name, value in metrics.items():
            self.log(f'train_{name}', value, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(
        self, 
        batch: Union[tuple, Dict[str, torch.Tensor]], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch de datos
            batch_idx: Índice del batch
            
        Returns:
            Validation loss
        """
        loss, metrics = self.shared_step(batch, batch_idx)
        
        # Logear loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Logear métricas
        for name, value in metrics.items():
            self.log(f'val_{name}', value, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(
        self, 
        batch: Union[tuple, Dict[str, torch.Tensor]], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Test step.
        
        Args:
            batch: Batch de datos
            batch_idx: Índice del batch
            
        Returns:
            Test loss
        """
        loss, metrics = self.shared_step(batch, batch_idx)
        
        # Logear loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Logear métricas
        for name, value in metrics.items():
            self.log(f'test_{name}', value, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Configura optimizador y scheduler.
        
        Returns:
            Optimizador o diccionario con optimizador y scheduler
        """
        if self.scheduler is None:
            return self.optimizer
        
        # Configuración completa con scheduler
        config = {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                **self.scheduler_config
            }
        }
        
        return config
