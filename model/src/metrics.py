"""
Métricas para segmentación binaria.

Este módulo proporciona métricas comúnmente usadas en segmentación semántica binaria,
optimizadas para PyTorch y compatibles con batches.

Todas las métricas:
- Aplican sigmoid automáticamente a las predicciones
- Soportan umbralización configurable
- Retornan el promedio sobre el batch
- Usan epsilon para estabilidad numérica
"""

import torch
from typing import Optional


class BinarySegmentationMetrics:
    """
    Clase base para métricas de segmentación binaria.
    
    Proporciona métodos comunes para calcular TP, FP, FN, TN y reducir
    duplicación de código.
    
    Args:
        threshold: Umbral para binarización de predicciones (default: 0.5)
        eps: Epsilon para estabilidad numérica (default: 1e-7)
    
    Example:
        >>> metrics = BinarySegmentationMetrics(threshold=0.5)
        >>> iou_value = metrics.iou(predictions, ground_truth)
        >>> f1_value = metrics.f1_score(predictions, ground_truth)
    """
    
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
    
    def _binarize(self, pr: torch.Tensor) -> torch.Tensor:
        """Aplica sigmoid y umbraliza las predicciones."""
        return torch.sigmoid(pr) > self.threshold
    
    def _compute_confusion_matrix(
        self, 
        pr: torch.Tensor, 
        gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula elementos de la matriz de confusión.
        
        Args:
            pr: Predicciones (logits)
            gt: Ground truth (binario)
            
        Returns:
            Tupla (tp, fp, fn, tn)
        """
        pr = self._binarize(pr)
        
        tp = torch.sum(gt * pr, dim=(-2, -1))
        fp = torch.sum(pr, dim=(-2, -1)) - tp
        fn = torch.sum(gt, dim=(-2, -1)) - tp
        tn = torch.sum((1 - gt.int()) * (1 - pr.int()), dim=(-2, -1))
        
        return tp, fp, fn, tn
    
    def iou(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Intersection over Union (IoU) / Jaccard Index.
        
        IoU = TP / (TP + FP + FN)
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            IoU promedio sobre el batch
        """
        pr = self._binarize(pr)
        
        intersection = torch.sum(gt * pr, dim=(-2, -1))
        union = (
            torch.sum(gt, dim=(-2, -1)) + 
            torch.sum(pr, dim=(-2, -1)) - 
            intersection + 
            self.eps
        )
        
        ious = (intersection + self.eps) / union
        return torch.mean(ious)
    
    def f1_score(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        F1 Score (Dice Coefficient).
        
        F1 = 2*TP / (2*TP + FP + FN)
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            F1 score promedio sobre el batch
        """
        tp, fp, fn, _ = self._compute_confusion_matrix(pr, gt)
        
        f1 = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        return torch.mean(f1)
    
    def precision(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Precision (Positive Predictive Value).
        
        Precision = TP / (TP + FP)
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            Precision promedio sobre el batch
        """
        tp, fp, _, _ = self._compute_confusion_matrix(pr, gt)
        
        prec = (tp + self.eps) / (tp + fp + self.eps)
        return torch.mean(prec)
    
    def recall(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Recall (Sensitivity, True Positive Rate).
        
        Recall = TP / (TP + FN)
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            Recall promedio sobre el batch
        """
        tp, _, fn, _ = self._compute_confusion_matrix(pr, gt)
        
        rec = (tp + self.eps) / (tp + fn + self.eps)
        return torch.mean(rec)
    
    def overall_accuracy(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Overall Accuracy (correctamente clasificados / total).
        
        OA = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            Accuracy promedio sobre el batch
        """
        tp, fp, fn, tn = self._compute_confusion_matrix(pr, gt)
        
        oa = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        return torch.mean(oa)
    
    def fw_iou(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Frequency Weighted IoU.
        
        FWIoU = Σ(freq_i * IoU_i) donde freq_i = (TP_i + FN_i) / total
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            FW-IoU promedio sobre el batch
        """
        tp, fp, fn, tn = self._compute_confusion_matrix(pr, gt)
        
        total = tp + fp + tn + fn
        freq = (tp + fn) / total
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        
        fw_iou_value = freq * iou
        return torch.mean(fw_iou_value)
    
    def kappa(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Cohen's Kappa coefficient.
        
        Kappa = (OA - PE) / (1 - PE)
        donde PE es el acuerdo esperado por azar.
        
        Args:
            pr: Predicciones (logits) [B, C, H, W]
            gt: Ground truth (binario) [B, C, H, W]
            
        Returns:
            Kappa promedio sobre el batch
        """
        tp, fp, fn, tn = self._compute_confusion_matrix(pr, gt)
        
        total = tp + tn + fp + fn
        oa = (tp + tn) / total
        
        # Probabilidad de acuerdo esperado por azar
        pe = (
            ((tp + fn) * (tp + fp)) + 
            ((fp + tn) * (fn + tn))
        ) / (total * total)
        
        k = (oa - pe) / (1 - pe + self.eps)
        return torch.mean(k)


# ============================================================================
# FUNCIONES STANDALONE (para compatibilidad con código existente)
# ============================================================================

# Instancia global con configuración por defecto
_default_metrics = BinarySegmentationMetrics(threshold=0.5, eps=1e-7)


def iou(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Intersection over Union (IoU).
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        IoU promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.iou(pr, gt)


def f1_score(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    F1 Score.
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        F1 score promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.f1_score(pr, gt)


def precision(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Precision.
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        Precision promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.precision(pr, gt)


def recall(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Recall.
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        Recall promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.recall(pr, gt)


def overall_accuracy(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Overall Accuracy.
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        Accuracy promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.overall_accuracy(pr, gt)


def fw_iou(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Frequency Weighted IoU.
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        FW-IoU promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.fw_iou(pr, gt)


def kappa(pr: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    Cohen's Kappa coefficient.
    
    Args:
        pr: Predicciones (logits)
        gt: Ground truth (binario)
        th: Umbral de binarización
        eps: Epsilon para estabilidad numérica
        
    Returns:
        Kappa promedio
    """
    metrics = BinarySegmentationMetrics(threshold=th, eps=eps)
    return metrics.kappa(pr, gt)
