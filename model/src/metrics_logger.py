"""
Callback para registro estructurado de métricas de entrenamiento.

Este módulo implementa un callback de PyTorch Lightning que captura y exporta
métricas de entrenamiento y validación en formato estructurado para publicación
científica.

Autor: Sistema de entrenamiento Multi-Branch UNet++
Fecha: 2026-01
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import lightning as L
from lightning.pytorch.callbacks import Callback


class MetricsHistoryCallback(Callback):
    """
    Callback personalizado para registro completo de métricas durante entrenamiento.

    Captura automáticamente:
    - Métricas de entrenamiento por época (loss, IoU, F1, precision, recall, etc.)
    - Métricas de validación por época
    - Metadata del experimento (configuración, timestamps, etc.)

    Al finalizar el entrenamiento, genera un archivo .txt estructurado con:
    - Información del experimento
    - Tabla de métricas por época (train y val)
    - Estadísticas de mejor modelo
    - Formato apropiado para citación en artículos científicos

    Args:
        output_dir: Directorio donde se guardará el archivo de métricas
        experiment_name: Nombre del experimento (para metadata)
        config: Objeto de configuración con parámetros del experimento

    Ejemplo de uso:
        >>> metrics_callback = MetricsHistoryCallback(
        ...     output_dir='outputs/metrics',
        ...     experiment_name='MultiBranch-UNetPP-ResNet34',
        ...     config=Config
        ... )
        >>> trainer = L.Trainer(callbacks=[metrics_callback])
    """

    def __init__(
        self,
        output_dir: str = 'outputs/metrics',
        experiment_name: str = 'experiment',
        config: Optional[object] = None
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.config = config

        # Crear directorio si no existe
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Almacenamiento de métricas por época
        self.train_metrics: Dict[str, List[float]] = {
            'loss': [],
            'OA': [],
            'iou': [],
            'f1': [],
            'pre': [],
            'recall': [],
            'FWIoU': [],
            'kp': []
        }

        self.val_metrics: Dict[str, List[float]] = {
            'loss': [],
            'OA': [],
            'iou': [],
            'f1': [],
            'pre': [],
            'recall': [],
            'FWIoU': [],
            'kp': []
        }

        # Metadata
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.best_epoch: int = 0
        self.total_epochs: int = 0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Registra el inicio del entrenamiento."""
        self.start_time = datetime.now()
        print(f"\n{'='*80}")
        print(f"INICIO DE REGISTRO DE MÉTRICAS")
        print(f"{'='*80}")
        print(f"📁 Directorio de salida: {self.output_dir}")
        print(f"📝 Experimento: {self.experiment_name}")
        print(f"{'='*80}\n")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Captura métricas de entrenamiento al final de cada época."""
        metrics = trainer.callback_metrics

        # Capturar pérdida de entrenamiento
        train_loss = metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.train_metrics['loss'].append(float(train_loss))

        # Capturar métricas de entrenamiento
        # Lightning registra métricas con prefijo 'train_'
        metric_keys = ['OA', 'iou', 'f1', 'pre', 'recall', 'FWIoU', 'kp']
        for key in metric_keys:
            train_key = f'train_{key}'
            if train_key in metrics:
                self.train_metrics[key].append(float(metrics[train_key]))

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Captura métricas de validación al final de cada época."""
        metrics = trainer.callback_metrics

        # Capturar pérdida de validación
        val_loss = metrics.get('val_loss')
        if val_loss is not None:
            self.val_metrics['loss'].append(float(val_loss))

        # Capturar métricas de validación
        # Lightning registra métricas con prefijo 'val_'
        metric_keys = ['OA', 'iou', 'f1', 'pre', 'recall', 'FWIoU', 'kp']
        for key in metric_keys:
            val_key = f'val_{key}'
            if val_key in metrics:
                self.val_metrics[key].append(float(metrics[val_key]))

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Al finalizar el entrenamiento, genera el archivo de métricas.
        """
        self.end_time = datetime.now()
        self.total_epochs = trainer.current_epoch + 1

        # Determinar mejor época (por val_iou)
        if self.val_metrics['iou']:
            self.best_epoch = self.val_metrics['iou'].index(max(self.val_metrics['iou'])) + 1

        # Generar archivo de métricas
        print(f"\n{'='*80}")
        print(f"GENERANDO ARCHIVO DE MÉTRICAS")
        print(f"{'='*80}")

        output_file = self._generate_metrics_file()

        print(f"✓ Archivo de métricas generado: {output_file}")
        print(f"{'='*80}\n")

    def _generate_metrics_file(self) -> Path:
        """
        Genera el archivo .txt con formato estructurado de métricas.

        Returns:
            Path del archivo generado
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_metrics_{self.experiment_name}_{timestamp}.txt'
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            # ================================================================
            # ENCABEZADO
            # ================================================================
            f.write("="*80 + "\n")
            f.write("REPORTE DE MÉTRICAS DE ENTRENAMIENTO\n")
            f.write("Multi-Branch UNet++ para Segmentación de Manglar\n")
            f.write("="*80 + "\n\n")

            # ================================================================
            # INFORMACIÓN DEL EXPERIMENTO
            # ================================================================
            f.write("INFORMACIÓN DEL EXPERIMENTO\n")
            f.write("-"*80 + "\n")
            f.write(f"Experimento:          {self.experiment_name}\n")
            f.write(f"Fecha de inicio:      {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Fecha de finalización: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            duration = self.end_time - self.start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            f.write(f"Duración total:       {int(hours)}h {int(minutes)}m {int(seconds)}s\n")

            f.write(f"Épocas completadas:   {self.total_epochs}\n")
            f.write(f"Mejor época:          {self.best_epoch} (val_iou máximo)\n")

            # Configuración del modelo (si está disponible)
            if self.config is not None:
                f.write("\nCONFIGURACIÓN DEL MODELO\n")
                f.write("-"*80 + "\n")
                f.write(f"Encoder:              {getattr(self.config, 'ENCODER', 'N/A')}\n")
                f.write(f"Fusion Mode:          {getattr(self.config, 'FUSION_MODE', 'N/A')}\n")
                f.write(f"Batch Size:           {getattr(self.config, 'BATCH_SIZE', 'N/A')}\n")
                f.write(f"Learning Rate:        {getattr(self.config, 'LR', 'N/A')}\n")
                f.write(f"Weight Decay:         {getattr(self.config, 'WEIGHT_DECAY', 'N/A')}\n")
                f.write(f"Deep Supervision:     {getattr(self.config, 'DEEP_SUPERVISION', 'N/A')}\n")

            f.write("\n" + "="*80 + "\n\n")

            # ================================================================
            # TABLA DE MÉTRICAS POR ÉPOCA
            # ================================================================
            f.write("MÉTRICAS POR ÉPOCA\n")
            f.write("="*80 + "\n\n")

            # ENCABEZADO DE LA TABLA
            f.write("Época | Train Loss | Val Loss   | Train IoU | Val IoU   | "
                   "Train F1  | Val F1    | Train Pre | Val Pre   | Train Rec | Val Rec   | "
                   "Train OA  | Val OA    | Train FWIoU | Val FWIoU | Train Kappa | Val Kappa\n")
            f.write("-"*80 + "\n")

            # FILAS DE DATOS
            num_epochs = len(self.val_metrics['loss'])

            for epoch in range(num_epochs):
                # Número de época
                line = f"{epoch+1:5d} | "

                # Pérdidas
                train_loss = self.train_metrics['loss'][epoch] if epoch < len(self.train_metrics['loss']) else 0.0
                val_loss = self.val_metrics['loss'][epoch]
                line += f"{train_loss:10.6f} | {val_loss:10.6f} | "

                # IoU
                train_iou = self.train_metrics['iou'][epoch] if epoch < len(self.train_metrics['iou']) else 0.0
                val_iou = self.val_metrics['iou'][epoch]
                line += f"{train_iou:9.6f} | {val_iou:9.6f} | "

                # F1
                train_f1 = self.train_metrics['f1'][epoch] if epoch < len(self.train_metrics['f1']) else 0.0
                val_f1 = self.val_metrics['f1'][epoch]
                line += f"{train_f1:9.6f} | {val_f1:9.6f} | "

                # Precision
                train_pre = self.train_metrics['pre'][epoch] if epoch < len(self.train_metrics['pre']) else 0.0
                val_pre = self.val_metrics['pre'][epoch]
                line += f"{train_pre:9.6f} | {val_pre:9.6f} | "

                # Recall
                train_rec = self.train_metrics['recall'][epoch] if epoch < len(self.train_metrics['recall']) else 0.0
                val_rec = self.val_metrics['recall'][epoch]
                line += f"{train_rec:9.6f} | {val_rec:9.6f} | "

                # Overall Accuracy
                train_oa = self.train_metrics['OA'][epoch] if epoch < len(self.train_metrics['OA']) else 0.0
                val_oa = self.val_metrics['OA'][epoch]
                line += f"{train_oa:9.6f} | {val_oa:9.6f} | "

                # FWIoU
                train_fwiou = self.train_metrics['FWIoU'][epoch] if epoch < len(self.train_metrics['FWIoU']) else 0.0
                val_fwiou = self.val_metrics['FWIoU'][epoch]
                line += f"{train_fwiou:11.6f} | {val_fwiou:11.6f} | "

                # Kappa
                train_kp = self.train_metrics['kp'][epoch] if epoch < len(self.train_metrics['kp']) else 0.0
                val_kp = self.val_metrics['kp'][epoch]
                line += f"{train_kp:11.6f} | {val_kp:11.6f}\n"

                # Marcar la mejor época
                if (epoch + 1) == self.best_epoch:
                    line = "* " + line  # Asterisco indica mejor época
                else:
                    line = "  " + line

                f.write(line)

            f.write("\n* Indica la mejor época según val_iou\n\n")

            # ================================================================
            # ESTADÍSTICAS DEL MEJOR MODELO
            # ================================================================
            f.write("="*80 + "\n")
            f.write("ESTADÍSTICAS DEL MEJOR MODELO\n")
            f.write("="*80 + "\n")

            best_idx = self.best_epoch - 1

            f.write(f"Mejor época:          {self.best_epoch}\n\n")

            f.write("MÉTRICAS DE ENTRENAMIENTO (Mejor época):\n")
            f.write("-"*80 + "\n")
            if best_idx < len(self.train_metrics['loss']):
                f.write(f"  Loss:               {self.train_metrics['loss'][best_idx]:.6f}\n")
                f.write(f"  IoU:                {self.train_metrics['iou'][best_idx]:.6f}\n")
                f.write(f"  F1 Score:           {self.train_metrics['f1'][best_idx]:.6f}\n")
                f.write(f"  Precision:          {self.train_metrics['pre'][best_idx]:.6f}\n")
                f.write(f"  Recall:             {self.train_metrics['recall'][best_idx]:.6f}\n")
                f.write(f"  Overall Accuracy:   {self.train_metrics['OA'][best_idx]:.6f}\n")
                f.write(f"  FW-IoU:             {self.train_metrics['FWIoU'][best_idx]:.6f}\n")
                f.write(f"  Kappa:              {self.train_metrics['kp'][best_idx]:.6f}\n")

            f.write("\nMÉTRICAS DE VALIDACIÓN (Mejor época):\n")
            f.write("-"*80 + "\n")
            f.write(f"  Loss:               {self.val_metrics['loss'][best_idx]:.6f}\n")
            f.write(f"  IoU:                {self.val_metrics['iou'][best_idx]:.6f}\n")
            f.write(f"  F1 Score:           {self.val_metrics['f1'][best_idx]:.6f}\n")
            f.write(f"  Precision:          {self.val_metrics['pre'][best_idx]:.6f}\n")
            f.write(f"  Recall:             {self.val_metrics['recall'][best_idx]:.6f}\n")
            f.write(f"  Overall Accuracy:   {self.val_metrics['OA'][best_idx]:.6f}\n")
            f.write(f"  FW-IoU:             {self.val_metrics['FWIoU'][best_idx]:.6f}\n")
            f.write(f"  Kappa:              {self.val_metrics['kp'][best_idx]:.6f}\n")

            # ================================================================
            # ESTADÍSTICAS GENERALES
            # ================================================================
            f.write("\n" + "="*80 + "\n")
            f.write("ESTADÍSTICAS GENERALES (VALIDACIÓN)\n")
            f.write("="*80 + "\n")

            f.write(f"Mejor val_iou:        {max(self.val_metrics['iou']):.6f} (época {self.best_epoch})\n")
            f.write(f"Peor val_iou:         {min(self.val_metrics['iou']):.6f}\n")
            f.write(f"Promedio val_iou:     {sum(self.val_metrics['iou'])/len(self.val_metrics['iou']):.6f}\n\n")

            f.write(f"Mejor val_f1:         {max(self.val_metrics['f1']):.6f}\n")
            f.write(f"Peor val_f1:          {min(self.val_metrics['f1']):.6f}\n")
            f.write(f"Promedio val_f1:      {sum(self.val_metrics['f1'])/len(self.val_metrics['f1']):.6f}\n\n")

            f.write(f"Menor val_loss:       {min(self.val_metrics['loss']):.6f}\n")
            f.write(f"Mayor val_loss:       {max(self.val_metrics['loss']):.6f}\n")
            f.write(f"Promedio val_loss:    {sum(self.val_metrics['loss'])/len(self.val_metrics['loss']):.6f}\n")

            # ================================================================
            # FORMATO PARA PUBLICACIÓN
            # ================================================================
            f.write("\n" + "="*80 + "\n")
            f.write("FORMATO PARA PUBLICACIÓN CIENTÍFICA\n")
            f.write("="*80 + "\n\n")

            f.write("Resultados del mejor modelo (Época {}):\n".format(self.best_epoch))
            f.write("-"*80 + "\n")
            f.write("Métrica           | Entrenamiento | Validación\n")
            f.write("-"*80 + "\n")

            if best_idx < len(self.train_metrics['loss']):
                f.write(f"Loss              | {self.train_metrics['loss'][best_idx]:13.6f} | "
                       f"{self.val_metrics['loss'][best_idx]:10.6f}\n")
                f.write(f"IoU               | {self.train_metrics['iou'][best_idx]:13.6f} | "
                       f"{self.val_metrics['iou'][best_idx]:10.6f}\n")
                f.write(f"F1 Score          | {self.train_metrics['f1'][best_idx]:13.6f} | "
                       f"{self.val_metrics['f1'][best_idx]:10.6f}\n")
                f.write(f"Precision         | {self.train_metrics['pre'][best_idx]:13.6f} | "
                       f"{self.val_metrics['pre'][best_idx]:10.6f}\n")
                f.write(f"Recall            | {self.train_metrics['recall'][best_idx]:13.6f} | "
                       f"{self.val_metrics['recall'][best_idx]:10.6f}\n")
                f.write(f"Overall Accuracy  | {self.train_metrics['OA'][best_idx]:13.6f} | "
                       f"{self.val_metrics['OA'][best_idx]:10.6f}\n")
                f.write(f"FW-IoU            | {self.train_metrics['FWIoU'][best_idx]:13.6f} | "
                       f"{self.val_metrics['FWIoU'][best_idx]:10.6f}\n")
                f.write(f"Kappa             | {self.train_metrics['kp'][best_idx]:13.6f} | "
                       f"{self.val_metrics['kp'][best_idx]:10.6f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("="*80 + "\n")

        return output_path

    def get_best_metrics(self) -> Dict[str, float]:
        """
        Retorna las métricas del mejor modelo como diccionario.

        Returns:
            Dict con métricas de entrenamiento y validación del mejor modelo
        """
        best_idx = self.best_epoch - 1

        return {
            'epoch': self.best_epoch,
            'train_loss': self.train_metrics['loss'][best_idx] if best_idx < len(self.train_metrics['loss']) else 0.0,
            'val_loss': self.val_metrics['loss'][best_idx],
            'train_iou': self.train_metrics['iou'][best_idx] if best_idx < len(self.train_metrics['iou']) else 0.0,
            'val_iou': self.val_metrics['iou'][best_idx],
            'train_f1': self.train_metrics['f1'][best_idx] if best_idx < len(self.train_metrics['f1']) else 0.0,
            'val_f1': self.val_metrics['f1'][best_idx],
            'train_precision': self.train_metrics['pre'][best_idx] if best_idx < len(self.train_metrics['pre']) else 0.0,
            'val_precision': self.val_metrics['pre'][best_idx],
            'train_recall': self.train_metrics['recall'][best_idx] if best_idx < len(self.train_metrics['recall']) else 0.0,
            'val_recall': self.val_metrics['recall'][best_idx],
            'train_oa': self.train_metrics['OA'][best_idx] if best_idx < len(self.train_metrics['OA']) else 0.0,
            'val_oa': self.val_metrics['OA'][best_idx],
            'train_fwiou': self.train_metrics['FWIoU'][best_idx] if best_idx < len(self.train_metrics['FWIoU']) else 0.0,
            'val_fwiou': self.val_metrics['FWIoU'][best_idx],
            'train_kappa': self.train_metrics['kp'][best_idx] if best_idx < len(self.train_metrics['kp']) else 0.0,
            'val_kappa': self.val_metrics['kp'][best_idx],
        }
