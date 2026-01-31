"""
SegFormer Wrapper para Segmentación de Manglares con Sentinel-2.

Este módulo proporciona un wrapper para SegFormer (Hugging Face Transformers)
adaptado para imágenes multiespectrales de Sentinel-2 con 6 bandas.

SegFormer es una arquitectura de Transformer eficiente para segmentación semántica
que combina un encoder jerárquico con un decoder MLP ligero.

Características principales:
- Encoder ViT jerárquico con atención eficiente (sin embeddings posicionales)
- Decoder MLP All-Layer que fusiona features multi-escala
- Adecuado para imágenes de baja resolución (128x128)
- Más eficiente que otros Vision Transformers

Referencias:
- Xie et al. (2021): "SegFormer: Simple and Efficient Design for Semantic Segmentation"
- https://huggingface.co/docs/transformers/model_doc/segformer

IMPORTANTE:
- SegFormer produce salidas de menor resolución (H/4, W/4) que se interpolan
- El modelo maneja 6 canales de entrada mediante adaptación de la capa inicial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from typing import Optional, Tuple


class SegFormerWrapper(nn.Module):
    """
    Wrapper de SegFormer para segmentación binaria con imágenes Sentinel-2.

    Adapta SegFormer de Hugging Face para:
    1. Aceptar 6 canales de entrada (en lugar de 3 RGB)
    2. Producir segmentación binaria (1 clase: manglar)
    3. Interpolar salida al tamaño original de entrada

    Args:
        model_name: Variante de SegFormer a usar
            - 'nvidia/segformer-b0-finetuned-ade-512-512' (más ligero, recomendado)
            - 'nvidia/segformer-b1-finetuned-ade-512-512' (balance rendimiento/eficiencia)
            - 'nvidia/segformer-b2-finetuned-ade-512-512' (mayor capacidad)
        in_channels: Número de canales de entrada (default: 6 para Sentinel-2)
        num_classes: Número de clases de salida (default: 1 para binario)
        pretrained: Si usar pesos preentrenados (default: True)
            - True: Carga pesos de ImageNet y adapta primera capa
            - False: Inicialización desde cero
        interpolate_output: Si interpolar salida al tamaño de entrada (default: True)
        output_hidden_states: Si retornar estados ocultos (para análisis)

    Ejemplo:
        >>> model = SegFormerWrapper(
        ...     model_name='nvidia/segformer-b0-finetuned-ade-512-512',
        ...     in_channels=6,
        ...     num_classes=1,
        ...     pretrained=True
        ... )
        >>> x = torch.randn(4, 6, 128, 128)  # Batch de imágenes Sentinel-2
        >>> y = model(x)  # Shape: [4, 1, 128, 128]

    DIFERENCIAS CON CNNs TRADICIONALES:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SegFormer               │         CNN (UNet++)    │
    ├────────────────────────────────────────────┼─────────────────────────┤
    │ Atención global desde el inicio            │ Receptive field local   │
    │ Mejor para contexto a larga distancia      │ Mejor para texturas     │
    │ Sin embeddings posicionales (flexible)     │ Tamaño fijo implícito   │
    │ Decoder MLP ligero                         │ Decoder convolucional   │
    │ Menos parámetros que ViT estándar          │ Más parámetros típicamente│
    └────────────────────────────────────────────┴─────────────────────────┘
    """

    # Mapeo de nombres amigables a modelos de Hugging Face
    MODEL_VARIANTS = {
        'segformer-b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'segformer-b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
        'segformer-b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'segformer-b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
        'segformer-b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
        'segformer-b5': 'nvidia/segformer-b5-finetuned-ade-512-512',
    }

    def __init__(
        self,
        model_name: str = 'segformer-b0',
        in_channels: int = 6,
        num_classes: int = 1,
        pretrained: bool = True,
        interpolate_output: bool = True,
        output_hidden_states: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.interpolate_output = interpolate_output
        self.output_hidden_states = output_hidden_states

        # Resolver nombre del modelo
        if model_name in self.MODEL_VARIANTS:
            hf_model_name = self.MODEL_VARIANTS[model_name]
        else:
            hf_model_name = model_name

        self.model_name = hf_model_name

        # Cargar configuración y modelo
        if pretrained:
            # Cargar modelo preentrenado y adaptar
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                hf_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,  # Permitir cambio de num_classes
            )

            # Adaptar la primera capa convolucional para 6 canales
            self._adapt_input_channels(in_channels)
        else:
            # Crear modelo desde configuración sin pesos preentrenados
            config = SegformerConfig.from_pretrained(hf_model_name)
            config.num_labels = num_classes
            config.num_channels = in_channels

            self.segformer = SegformerForSemanticSegmentation(config)

        # Información del modelo
        self._print_model_info()

    def _adapt_input_channels(self, in_channels: int):
        """
        Adapta la primera capa convolucional para aceptar más canales.

        Estrategia: Promediar pesos de los 3 canales RGB y replicar para
        los canales adicionales. Esto preserva los patrones aprendidos
        mientras extiende la capacidad de entrada.

        Args:
            in_channels: Número de canales de entrada deseados
        """
        # La primera capa está en el encoder (patch embeddings)
        # SegFormer usa patch_embeddings en cada etapa

        # Acceder a la primera capa de patch embedding
        first_conv = self.segformer.segformer.encoder.patch_embeddings[0].proj

        if first_conv.in_channels == in_channels:
            return  # No necesita adaptación

        # Obtener pesos originales [out_channels, 3, kernel_h, kernel_w]
        original_weight = first_conv.weight.data
        out_channels = original_weight.shape[0]
        kernel_size = (original_weight.shape[2], original_weight.shape[3])
        stride = first_conv.stride
        padding = first_conv.padding

        # Crear nueva capa convolucional con más canales de entrada
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=first_conv.bias is not None
        )

        # Inicializar pesos de la nueva capa
        with torch.no_grad():
            # Promediar pesos de los 3 canales RGB
            rgb_mean = original_weight.mean(dim=1, keepdim=True)  # [out, 1, kh, kw]

            # Crear nuevo tensor de pesos
            new_weight = torch.zeros(out_channels, in_channels, *kernel_size)

            # Copiar pesos originales para los primeros 3 canales (aproximadamente RGB)
            # Para Sentinel-2: B2 (Blue), B3 (Green), B4 (Red)
            new_weight[:, :3, :, :] = original_weight

            # Para canales adicionales (NIR, SWIR1, SWIR2), usar promedio
            for i in range(3, in_channels):
                new_weight[:, i, :, :] = rgb_mean.squeeze(1)

            # Aplicar escalado para mantener magnitud similar
            # (dado que tenemos más canales de entrada)
            scale_factor = 3.0 / in_channels
            new_weight *= scale_factor

            new_conv.weight.data = new_weight

            # Copiar bias si existe
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        # Reemplazar la capa
        self.segformer.segformer.encoder.patch_embeddings[0].proj = new_conv

    def _print_model_info(self):
        """Imprime información del modelo."""
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("SegFormer Wrapper Inicializado")
        print("=" * 70)
        print(f"Modelo base: {self.model_name}")
        print(f"Canales de entrada: {self.in_channels}")
        print(f"Clases de salida: {self.num_classes}")
        print(f"Interpolación a tamaño original: {self.interpolate_output}")
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print("=" * 70 + "\n")

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada con shape [B, C, H, W]
               donde C = in_channels (6 para Sentinel-2)
            return_dict: Si retornar diccionario con información adicional

        Returns:
            Si return_dict=False:
                Tensor de logits con shape [B, num_classes, H, W]
            Si return_dict=True:
                Diccionario con 'logits', 'hidden_states' (si aplica)

        Nota:
            SegFormer produce salidas de menor resolución (H/4, W/4).
            Si interpolate_output=True, se interpola de vuelta a (H, W).
        """
        input_shape = x.shape[-2:]  # (H, W)

        # Forward a través de SegFormer
        outputs = self.segformer(
            pixel_values=x,
            output_hidden_states=self.output_hidden_states,
            return_dict=True
        )

        # Obtener logits [B, num_classes, H/4, W/4]
        logits = outputs.logits

        # Interpolar al tamaño original si es necesario
        if self.interpolate_output and logits.shape[-2:] != input_shape:
            logits = F.interpolate(
                logits,
                size=input_shape,
                mode='bilinear',
                align_corners=False
            )

        if return_dict:
            result = {'logits': logits}
            if self.output_hidden_states:
                result['hidden_states'] = outputs.hidden_states
            return result

        return logits


class SegFormerWithAuxLoss(SegFormerWrapper):
    """
    SegFormer con cabeza auxiliar para supervisión profunda.

    Agrega una cabeza auxiliar en una etapa intermedia del encoder
    para mejorar el flujo de gradientes durante el entrenamiento.

    Similar a la estrategia de deep supervision en UNet++.

    Args:
        aux_stage: Etapa del encoder para la salida auxiliar (0-3)
        aux_weight: Peso de la pérdida auxiliar en el total
        **kwargs: Argumentos para SegFormerWrapper
    """

    def __init__(
        self,
        aux_stage: int = 2,
        aux_weight: float = 0.4,
        **kwargs
    ):
        # Forzar output_hidden_states para tener acceso a estados intermedios
        kwargs['output_hidden_states'] = True
        super().__init__(**kwargs)

        self.aux_stage = aux_stage
        self.aux_weight = aux_weight

        # Obtener dimensión de features de la etapa auxiliar
        # Las dimensiones varían según la variante (B0-B5)
        config = self.segformer.config
        aux_channels = config.hidden_sizes[aux_stage]

        # Cabeza auxiliar simple: Conv 1x1 para reducir canales
        self.aux_head = nn.Sequential(
            nn.Conv2d(aux_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, kwargs.get('num_classes', 1), kernel_size=1)
        )

        print(f"Cabeza auxiliar en etapa {aux_stage} (channels={aux_channels})")
        print(f"Peso de pérdida auxiliar: {aux_weight}")

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward con salida auxiliar opcional.

        Args:
            x: Tensor de entrada [B, C, H, W]
            return_aux: Si retornar salida auxiliar

        Returns:
            Tupla (logits_main, logits_aux) si return_aux=True
            Solo logits_main si return_aux=False
        """
        input_shape = x.shape[-2:]

        # Forward completo
        outputs = self.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )

        # Salida principal
        logits = outputs.logits
        if self.interpolate_output and logits.shape[-2:] != input_shape:
            logits = F.interpolate(
                logits,
                size=input_shape,
                mode='bilinear',
                align_corners=False
            )

        if not return_aux:
            return logits

        # Salida auxiliar
        hidden_states = outputs.hidden_states
        aux_features = hidden_states[self.aux_stage]
        aux_logits = self.aux_head(aux_features)

        # Interpolar auxiliar al tamaño original
        if aux_logits.shape[-2:] != input_shape:
            aux_logits = F.interpolate(
                aux_logits,
                size=input_shape,
                mode='bilinear',
                align_corners=False
            )

        return logits, aux_logits


def create_segformer_model(
    variant: str = 'b0',
    in_channels: int = 6,
    num_classes: int = 1,
    pretrained: bool = True,
    use_aux_loss: bool = False,
    aux_weight: float = 0.4
) -> nn.Module:
    """
    Factory function para crear modelos SegFormer.

    Args:
        variant: Variante del modelo ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
        in_channels: Canales de entrada (6 para Sentinel-2)
        num_classes: Clases de segmentación (1 para binario)
        pretrained: Usar pesos preentrenados de ImageNet
        use_aux_loss: Agregar cabeza auxiliar para deep supervision
        aux_weight: Peso de la pérdida auxiliar

    Returns:
        Modelo SegFormer configurado

    Recomendaciones por tamaño de tesela:
    ┌────────────────┬─────────────────────────────────────────────┐
    │ Tamaño tesela  │ Variante recomendada                        │
    ├────────────────┼─────────────────────────────────────────────┤
    │ 64x64          │ B0 (menor overhead, patches más pequeños)   │
    │ 128x128        │ B0 o B1 (balance rendimiento/eficiencia)    │
    │ 256x256        │ B1 o B2 (mayor capacidad justificada)       │
    │ 512x512+       │ B2-B5 (arquitecturas más grandes)           │
    └────────────────┴─────────────────────────────────────────────┘
    """
    model_name = f'segformer-{variant}'

    if use_aux_loss:
        return SegFormerWithAuxLoss(
            model_name=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            aux_weight=aux_weight,
        )
    else:
        return SegFormerWrapper(
            model_name=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
        )


# =============================================================================
# CONFIGURACIÓN RECOMENDADA PARA MANGLARES (128x128)
# =============================================================================
"""
RECOMENDACIONES DE HIPERPARÁMETROS PARA SEGFORMER CON TESELAS 128x128:

1. ARQUITECTURA:
   - Variante: B0 o B1 (suficiente para 128x128, mayor es overkill)
   - B0: ~3.7M params, más rápido
   - B1: ~13.7M params, mejor precisión potencial

2. LEARNING RATE:
   - Base: 6e-5 a 2e-4 (Transformers requieren LR menor que CNNs)
   - Con AdamW: LR=6e-5, weight_decay=0.01
   - Warmup: 10% de épocas totales

3. BATCH SIZE:
   - Teselas 128x128 con B0: batch_size=32-64 (según GPU)
   - Teselas 128x128 con B1: batch_size=16-32

4. DATA AUGMENTATION:
   - SegFormer se beneficia de augmentación similar a CNNs
   - RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
   - RandAugment o AutoAugment pueden ayudar

5. SCHEDULER:
   - Polynomial decay (usado en paper original)
   - O ReduceLROnPlateau con patience=10

6. ÉPOCAS:
   - 100-200 épocas típicamente
   - Early stopping con patience=20-30

7. MANEJO DE DESBALANCE:
   - Dice + Focal Loss (igual que CNNs)
   - class_weight en CrossEntropy si se usa

8. REGULARIZACIÓN:
   - SegFormer ya tiene dropout interno
   - Weight decay=0.01 típico
   - Label smoothing=0.1 puede ayudar

COMPARACIÓN ESPERADA CON CNNs:
- SegFormer puede superar CNNs en teselas con manglar disperso
  gracias a su capacidad de capturar contexto global
- En teselas con manglar denso, diferencias menores
- Tiempo de inferencia similar o menor que UNet++ con ResNet50+
"""
