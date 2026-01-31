"""
Mask2Former Wrapper para Segmentación de Manglares con Sentinel-2.

Mask2Former es una arquitectura unificada estado del arte para segmentación
panóptica, de instancias y semántica. Utiliza masked attention para mejorar
la convergencia y precisión.

Características principales:
- Masked cross-attention que restringe la atención a regiones relevantes
- Multi-scale deformable attention en el decoder
- Estado del arte en múltiples benchmarks (ADE20K, COCO, Cityscapes)
- Backbone flexible (Swin Transformer, ResNet)

Referencias:
- Cheng et al. (2022): "Masked-attention Mask Transformer for Universal
  Image Segmentation" (NeurIPS 2022)
- https://huggingface.co/docs/transformers/model_doc/mask2former

RENDIMIENTO REPORTADO:
- ADE20K: 57.7 mIoU (semántico)
- COCO: 57.8 PQ (panóptico), 50.1 AP (instancias)
- Cityscapes: 83.3 mIoU

NOTA IMPORTANTE:
Mask2Former en HuggingFace actualmente solo soporta Swin Transformer como backbone.
Para imágenes pequeñas (128x128), se recomienda usar variantes tiny o small.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    AutoImageProcessor
)
from typing import Optional, Dict, Any
import warnings


class Mask2FormerWrapper(nn.Module):
    """
    Wrapper de Mask2Former para segmentación semántica binaria con Sentinel-2.

    Adapta Mask2Former de Hugging Face para:
    1. Aceptar 6 canales de entrada (Sentinel-2)
    2. Producir segmentación semántica binaria (1 clase: manglar)
    3. Manejar imágenes pequeñas (128x128)

    Args:
        model_name: Nombre del modelo preentrenado de HuggingFace
            - 'facebook/mask2former-swin-tiny-ade-semantic' (recomendado para 128x128)
            - 'facebook/mask2former-swin-small-ade-semantic'
            - 'facebook/mask2former-swin-base-ade-semantic'
            - 'facebook/mask2former-swin-large-ade-semantic'
        in_channels: Número de canales de entrada (6 para Sentinel-2)
        num_classes: Número de clases (1 para binario, 2 para multiclase)
        pretrained: Si usar pesos preentrenados
        ignore_mismatched_sizes: Ignorar discrepancias de tamaño al cargar pesos

    Ejemplo:
        >>> model = Mask2FormerWrapper(
        ...     model_name='facebook/mask2former-swin-tiny-ade-semantic',
        ...     in_channels=6,
        ...     num_classes=1
        ... )
        >>> x = torch.randn(4, 6, 128, 128)
        >>> y = model(x)  # Shape: [4, 1, 128, 128]

    DIFERENCIAS CON OTROS MODELOS:
    ┌────────────────────────────────────────────────────────────────────────┐
    │              Mask2Former              │         SegFormer              │
    ├──────────────────────────────────────┼─────────────────────────────────┤
    │ Masked cross-attention               │ Self-attention estándar         │
    │ Queries para máscaras                │ Decoder MLP simple              │
    │ Multi-scale deformable attention     │ Fusión multi-escala MLP         │
    │ Más parámetros, mayor precisión      │ Más ligero, buena eficiencia    │
    │ Mejor para bordes precisos           │ Mejor balance velocidad/calidad │
    └──────────────────────────────────────┴─────────────────────────────────┘
    """

    # Modelos disponibles en HuggingFace
    MODEL_VARIANTS = {
        'mask2former-tiny': 'facebook/mask2former-swin-tiny-ade-semantic',
        'mask2former-small': 'facebook/mask2former-swin-small-ade-semantic',
        'mask2former-base': 'facebook/mask2former-swin-base-ade-semantic',
        'mask2former-large': 'facebook/mask2former-swin-large-ade-semantic',
    }

    def __init__(
        self,
        model_name: str = 'mask2former-tiny',
        in_channels: int = 6,
        num_classes: int = 1,
        pretrained: bool = True,
        ignore_mismatched_sizes: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Resolver nombre del modelo
        if model_name in self.MODEL_VARIANTS:
            hf_model_name = self.MODEL_VARIANTS[model_name]
        else:
            hf_model_name = model_name

        self.model_name = hf_model_name

        # Para segmentación binaria, usamos 2 clases (background + foreground)
        # o 1 clase con sigmoid
        effective_num_classes = num_classes if num_classes > 1 else 2

        if pretrained:
            # Cargar modelo preentrenado
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                hf_model_name,
                num_labels=effective_num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes
            )
        else:
            # Crear desde configuración
            config = Mask2FormerConfig.from_pretrained(hf_model_name)
            config.num_labels = effective_num_classes
            self.model = Mask2FormerForUniversalSegmentation(config)

        # Adaptar primera capa para 6 canales
        self._adapt_input_channels(in_channels)

        # Capa de proyección final para binario
        if num_classes == 1:
            self.final_conv = nn.Conv2d(effective_num_classes, 1, kernel_size=1)
        else:
            self.final_conv = None

        # Imprimir información
        self._print_model_info()

    def _adapt_input_channels(self, in_channels: int):
        """
        Adapta la primera capa para aceptar 6 canales de entrada.

        La adaptación se hace en el patch embedding del Swin Transformer backbone.
        """
        # Acceder al backbone (Swin Transformer)
        backbone = self.model.model.pixel_level_module.encoder

        # La primera capa de Swin está en patch_embed
        if hasattr(backbone, 'embeddings'):
            patch_embed = backbone.embeddings
            if hasattr(patch_embed, 'patch_embeddings'):
                first_conv = patch_embed.patch_embeddings.projection

                if first_conv.in_channels == in_channels:
                    return  # No necesita adaptación

                # Crear nueva convolución
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )

                # Inicializar pesos
                with torch.no_grad():
                    original_weight = first_conv.weight.data
                    # Promediar canales RGB
                    rgb_mean = original_weight.mean(dim=1, keepdim=True)

                    # Crear nuevos pesos
                    new_weight = torch.zeros(
                        first_conv.out_channels,
                        in_channels,
                        *first_conv.kernel_size
                    )

                    # Copiar RGB
                    new_weight[:, :3, :, :] = original_weight
                    # Extender a canales adicionales
                    for i in range(3, in_channels):
                        new_weight[:, i:i+1, :, :] = rgb_mean

                    # Escalar
                    new_weight *= 3.0 / in_channels
                    new_conv.weight.data = new_weight

                    if first_conv.bias is not None:
                        new_conv.bias.data = first_conv.bias.data.clone()

                # Reemplazar
                patch_embed.patch_embeddings.projection = new_conv

    def _print_model_info(self):
        """Imprime información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("Mask2Former Wrapper Inicializado")
        print("=" * 70)
        print(f"Modelo base: {self.model_name}")
        print(f"Canales de entrada: {self.in_channels}")
        print(f"Clases de salida: {self.num_classes}")
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print("=" * 70 + "\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada [B, C, H, W] donde C = in_channels

        Returns:
            Tensor de logits [B, num_classes, H, W]
        """
        input_shape = x.shape[-2:]  # (H, W)

        # Forward de Mask2Former
        outputs = self.model(pixel_values=x)

        # Obtener máscaras y clasificaciones de queries
        # masks_queries_logits: [B, num_queries, H/4, W/4]
        # class_queries_logits: [B, num_queries, num_classes+1] (última clase = "no object")
        masks_queries_logits = outputs.masks_queries_logits
        class_queries_logits = outputs.class_queries_logits

        batch_size = x.shape[0]
        num_queries = masks_queries_logits.shape[1]
        mask_h, mask_w = masks_queries_logits.shape[-2:]

        # Interpolar máscaras al tamaño original PRIMERO (antes de softmax)
        masks_queries_logits = F.interpolate(
            masks_queries_logits,
            size=input_shape,
            mode='bilinear',
            align_corners=False
        )

        # Para segmentación semántica, usamos el enfoque de HuggingFace:
        # 1. Para cada píxel, encontrar la query con mayor score
        # 2. Asignar la clase de esa query al píxel

        # Excluir clase "no object" (última clase) para clasificación
        # class_queries_logits shape: [B, Q, num_classes+1]
        class_logits = class_queries_logits[..., :-1]  # [B, Q, num_classes]

        # Para segmentación binaria (2 clases: background, foreground)
        # Queremos obtener logits para la clase foreground (clase 1)
        num_classes = class_logits.shape[-1]

        # Calcular scores combinados: mask_score + class_score
        # Esto permite competencia entre queries para cada píxel
        # masks_queries_logits: [B, Q, H, W]
        # class_logits: [B, Q, C]

        # Expandir class_logits para broadcasting
        # [B, Q, C] -> [B, Q, C, 1, 1]
        class_logits_expanded = class_logits.unsqueeze(-1).unsqueeze(-1)

        # Expandir masks para broadcasting
        # [B, Q, H, W] -> [B, Q, 1, H, W]
        masks_expanded = masks_queries_logits.unsqueeze(2)

        # Scores combinados: [B, Q, C, H, W]
        combined_scores = masks_expanded + class_logits_expanded

        # Para cada píxel y clase, tomar el máximo score entre queries
        # [B, Q, C, H, W] -> [B, C, H, W] (max over Q dimension)
        semantic_logits, _ = combined_scores.max(dim=1)

        # Para binario, tomar la diferencia foreground - background
        # o usar solo foreground si num_classes == 2
        if self.num_classes == 1:
            if num_classes >= 2:
                # Logits = foreground_score - background_score (como logit ratio)
                semantic_logits = semantic_logits[:, 1:2, :, :] - semantic_logits[:, 0:1, :, :]
            else:
                # Solo una clase, usar directamente
                semantic_logits = semantic_logits[:, 0:1, :, :]
        elif self.final_conv is not None:
            semantic_logits = self.final_conv(semantic_logits)

        return semantic_logits


def create_mask2former_model(
    variant: str = 'tiny',
    in_channels: int = 6,
    num_classes: int = 1,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function para crear modelos Mask2Former.

    Args:
        variant: Variante del modelo ('tiny', 'small', 'base', 'large')
        in_channels: Canales de entrada
        num_classes: Clases de segmentación
        pretrained: Usar pesos preentrenados

    Returns:
        Modelo Mask2Former configurado

    Recomendaciones:
    ┌────────────────┬──────────────────────────────────────────────┐
    │ Tamaño tesela  │ Variante recomendada                         │
    ├────────────────┼──────────────────────────────────────────────┤
    │ 128x128        │ tiny (menor overhead, más rápido)            │
    │ 256x256        │ small o base                                 │
    │ 512x512+       │ base o large                                 │
    └────────────────┴──────────────────────────────────────────────┘
    """
    model_name = f'mask2former-{variant}'

    return Mask2FormerWrapper(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained
    )


# =============================================================================
# INFORMACIÓN ADICIONAL
# =============================================================================
"""
COMPARACIÓN DE RENDIMIENTO ESPERADO vs OTROS MODELOS:

Para segmentación de manglares (128x128):
┌─────────────────┬────────────────┬─────────────────────────────────────────┐
│ Modelo          │ Params aprox.  │ Características                         │
├─────────────────┼────────────────┼─────────────────────────────────────────┤
│ Mask2Former-T   │ ~47M           │ SOTA, bordes precisos, más lento        │
│ SegFormer-B0    │ ~3.7M          │ Muy eficiente, buen balance             │
│ SegFormer-B1    │ ~13.7M         │ Mejor precisión, aún eficiente          │
│ UNet++          │ ~26M (ResNet50)│ Robusto, skip connections densas        │
│ DeepLabV3+      │ ~41M (ResNet101)│ ASPP para contexto multi-escala        │
│ SegNet-Lite     │ ~7M            │ Clásico, baseline histórico             │
└─────────────────┴────────────────┴─────────────────────────────────────────┘

CUÁNDO USAR MASK2FORMER:
1. Cuando se necesita máxima precisión en bordes
2. Para segmentación de objetos con formas complejas
3. Cuando el tiempo de entrenamiento no es crítico
4. Para comparar con estado del arte absoluto

CUÁNDO NO USAR MASK2FORMER:
1. Recursos de GPU limitados (requiere más memoria)
2. Cuando se necesita inferencia muy rápida
3. Para imágenes muy pequeñas (puede ser overkill)
"""
