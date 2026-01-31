"""
SegNet: Arquitectura Encoder-Decoder para Segmentación Semántica.

SegNet es una arquitectura clásica de segmentación que utiliza:
- Encoder basado en VGG16 (capas convolucionales)
- Decoder simétrico con unpooling usando índices de max pooling
- Sin skip connections (a diferencia de UNet)

Características principales:
- Memoria eficiente: solo almacena índices de pooling, no feature maps completos
- Decoder simétrico al encoder
- Arquitectura más simple que UNet

Referencias:
- Badrinarayanan et al. (2017): "SegNet: A Deep Convolutional Encoder-Decoder
  Architecture for Image Segmentation"

DIFERENCIAS CON OTRAS ARQUITECTURAS:
┌─────────────────────────────────────────────────────────────────────────┐
│                 SegNet                  │         UNet                  │
├─────────────────────────────────────────┼───────────────────────────────┤
│ Sin skip connections                    │ Skip connections densas       │
│ Unpooling con índices                   │ Upsampling + concatenación    │
│ Menor uso de memoria                    │ Mayor uso de memoria          │
│ Bordes menos precisos                   │ Mejor preservación de bordes  │
│ Arquitectura más simple                 │ Más compleja pero efectiva    │
└─────────────────────────────────────────┴───────────────────────────────┘

Nota: SegNet históricamente tiene menor rendimiento que UNet en la mayoría
de benchmarks, pero se incluye para completitud en la comparación.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBNReLU(nn.Module):
    """Bloque Convolución + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """
    Bloque encoder de SegNet.

    Aplica N convoluciones seguidas de max pooling con retorno de índices.
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()

        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ConvBNReLU(in_ch, out_channels))

        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """
        Forward pass.

        Returns:
            x: Feature map después de pooling
            indices: Índices de max pooling (para unpooling)
            size: Tamaño antes de pooling (para unpooling)
        """
        x = self.convs(x)
        size = x.size()
        x, indices = self.pool(x)
        return x, indices, size


class DecoderBlock(nn.Module):
    """
    Bloque decoder de SegNet.

    Aplica unpooling usando índices seguido de N convoluciones.
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            # Última conv del bloque va a out_channels
            out_ch = out_channels
            layers.append(ConvBNReLU(in_ch, out_ch))

        self.convs = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        output_size: torch.Size
    ) -> torch.Tensor:
        """
        Forward pass con unpooling.

        Args:
            x: Feature map de entrada
            indices: Índices de max pooling del encoder correspondiente
            output_size: Tamaño de salida esperado
        """
        x = self.unpool(x, indices, output_size=output_size)
        x = self.convs(x)
        return x


class SegNet(nn.Module):
    """
    SegNet: Encoder-Decoder con unpooling basado en índices.

    Arquitectura inspirada en VGG16:
    - Encoder: 5 bloques con 2-3 convoluciones cada uno
    - Decoder: 5 bloques simétricos con unpooling

    Args:
        in_channels: Número de canales de entrada (6 para Sentinel-2)
        num_classes: Número de clases de salida (1 para binario)
        encoder_channels: Lista de canales para cada bloque encoder
        init_weights: Si inicializar pesos con estrategia específica

    Ejemplo:
        >>> model = SegNet(in_channels=6, num_classes=1)
        >>> x = torch.randn(4, 6, 128, 128)
        >>> y = model(x)  # Shape: [4, 1, 128, 128]
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        encoder_channels: List[int] = None,
        init_weights: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Canales por defecto (inspirado en VGG16)
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 512]

        self.encoder_channels = encoder_channels

        # Número de convoluciones por bloque (VGG16 style)
        num_convs = [2, 2, 3, 3, 3]

        # ============== ENCODER ==============
        self.encoders = nn.ModuleList()
        prev_channels = in_channels

        for i, (out_ch, n_conv) in enumerate(zip(encoder_channels, num_convs)):
            self.encoders.append(EncoderBlock(prev_channels, out_ch, n_conv))
            prev_channels = out_ch

        # ============== DECODER ==============
        # Decoder es simétrico al encoder (en orden inverso)
        decoder_channels = encoder_channels[::-1]  # Invertir
        decoder_convs = num_convs[::-1]

        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            # El último decoder va a encoder_channels[0], los demás al siguiente nivel
            out_ch = decoder_channels[i + 1] if i < len(decoder_channels) - 1 else encoder_channels[0]
            n_conv = decoder_convs[i]
            self.decoders.append(DecoderBlock(in_ch, out_ch, n_conv))

        # ============== CLASSIFIER ==============
        self.classifier = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

        # Inicialización de pesos
        if init_weights:
            self._initialize_weights()

        # Imprimir información
        self._print_model_info()

    def _initialize_weights(self):
        """Inicializa pesos con estrategia Kaiming."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        """Imprime información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("SegNet Inicializado")
        print("=" * 70)
        print(f"Canales de entrada: {self.in_channels}")
        print(f"Clases de salida: {self.num_classes}")
        print(f"Canales encoder: {self.encoder_channels}")
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print("=" * 70 + "\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de SegNet.

        Args:
            x: Tensor de entrada [B, C, H, W]

        Returns:
            Logits de segmentación [B, num_classes, H, W]
        """
        # Almacenar índices y tamaños para el decoder
        indices_list = []
        sizes_list = []

        # ============== ENCODER ==============
        for encoder in self.encoders:
            x, indices, size = encoder(x)
            indices_list.append(indices)
            sizes_list.append(size)

        # ============== DECODER ==============
        # Invertir listas para hacer matching con decoder
        indices_list = indices_list[::-1]
        sizes_list = sizes_list[::-1]

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, indices_list[i], sizes_list[i])

        # ============== CLASSIFIER ==============
        x = self.classifier(x)

        return x


class SegNetLite(nn.Module):
    """
    Versión ligera de SegNet para imágenes pequeñas (128x128).

    Reduce el número de canales y bloques para menor memoria y
    entrenamiento más rápido.

    Args:
        in_channels: Canales de entrada
        num_classes: Clases de salida
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1
    ):
        super().__init__()

        # Canales reducidos para imágenes pequeñas
        encoder_channels = [32, 64, 128, 256, 256]

        self.segnet = SegNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            init_weights=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segnet(x)


def create_segnet_model(
    variant: str = 'standard',
    in_channels: int = 6,
    num_classes: int = 1
) -> nn.Module:
    """
    Factory function para crear modelos SegNet.

    Args:
        variant: Variante del modelo
            - 'standard': SegNet completo (VGG16-style)
            - 'lite': Versión ligera para imágenes pequeñas
        in_channels: Canales de entrada
        num_classes: Clases de segmentación

    Returns:
        Modelo SegNet configurado
    """
    if variant == 'lite':
        return SegNetLite(in_channels=in_channels, num_classes=num_classes)
    else:
        return SegNet(in_channels=in_channels, num_classes=num_classes)


# =============================================================================
# INFORMACIÓN ADICIONAL
# =============================================================================
"""
COMPARACIÓN DE RENDIMIENTO ESPERADO:

SegNet vs UNet (en la mayoría de benchmarks):
- SegNet: IoU típico 5-10% menor que UNet
- SegNet: Menor uso de memoria (~50% menos)
- SegNet: Bordes menos definidos (sin skip connections)

CUÁNDO USAR SEGNET:
1. Recursos limitados de memoria
2. Comparación histórica con arquitecturas clásicas
3. Cuando los bordes precisos no son críticos

CUÁNDO NO USAR SEGNET:
1. Cuando se necesita máxima precisión
2. Para segmentación de objetos pequeños (manglar disperso)
3. Cuando la memoria no es una limitación

RECOMENDACIÓN PARA MANGLARES:
- SegNet probablemente tendrá peor rendimiento que UNet/SegFormer
- en teselas con manglar disperso debido a la ausencia de skip connections
- Incluido principalmente para completitud en la comparación de arquitecturas
"""
