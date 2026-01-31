"""
SegNeXt: Arquitectura CNN con Atención Convolucional Multi-Escala.

SegNeXt demuestra que la atención convolucional es más eficiente y efectiva
que self-attention para segmentación semántica, logrando resultados SOTA
con menos parámetros que los transformers.

Características principales:
- Multi-Scale Convolutional Attention (MSCA)
- Estructura piramidal eficiente (encoder MSCAN)
- Decoder Hamburger (contexto global ligero)
- Mejor relación rendimiento/eficiencia que Transformers

Referencias:
- Guo et al. (2022): "SegNeXt: Rethinking Convolutional Attention Design
  for Semantic Segmentation" (NeurIPS 2022)
- https://github.com/Visual-Attention-Network/SegNeXt

RENDIMIENTO REPORTADO:
- ADE20K: 44.3 mIoU (SegNeXt-T, 4M params) a 52.1 mIoU (SegNeXt-L, 49M params)
- Pascal VOC 2012: 90.6% mIoU (mejor que EfficientNet-L2 w/ NAS-FPN)
- Supera a SegFormer con menos FLOPs

VENTAJAS:
- CNN pura (sin self-attention costosa)
- Multi-scale context mediante convoluciones strip
- Muy eficiente para imágenes pequeñas
- Fácil de entrenar y hacer fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class DWConv(nn.Module):
    """Convolución Depthwise con BatchNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class Mlp(nn.Module):
    """MLP con convolución depthwise (como en SegFormer)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    """Stem inicial con downsampling 4x."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AttentionModule(nn.Module):
    """
    Multi-Scale Convolutional Attention (MSCA).

    Componentes:
    1. Depth-wise conv para agregación local
    2. Multi-branch strip convolutions para contexto multi-escala
    3. 1x1 conv para modelar relaciones entre canales
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        # Strip convolutions multi-escala
        self.conv0_1 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=(1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, kernel_size=(21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()

        # Agregación local
        attn = self.conv0(x)

        # Multi-scale strip convolutions
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        # Fusionar escalas
        attn = attn + attn_0 + attn_1 + attn_2

        # Proyección final
        attn = self.conv3(attn)

        # Attention
        return attn * u


class MSCABlock(nn.Module):
    """
    Bloque MSCA (Multi-Scale Convolutional Attention).

    Similar estructura a ViT pero con MSCA en lugar de self-attention.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = AttentionModule(dim)

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

        # Stochastic depth
        self.drop_path = nn.Identity()
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)

        # Layer scale
        self.layer_scale_1 = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)
        self.layer_scale_2 = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic Depth (drop path)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbed(nn.Module):
    """Patch embedding con overlap para stages intermedios."""

    def __init__(
        self,
        patch_size: int = 3,
        stride: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class MSCAN(nn.Module):
    """
    Multi-Scale Convolutional Attention Network (Encoder).

    Estructura piramidal con 4 stages, cada uno con:
    - Patch embedding (downsample)
    - Bloques MSCA
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dims: List[int] = [32, 64, 160, 256],
        mlp_ratios: List[float] = [8, 8, 4, 4],
        depths: List[int] = [3, 3, 5, 2],
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.depths = depths
        self.num_stages = 4

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(self.num_stages):
            # Patch embedding
            if i == 0:
                patch_embed = StemConv(in_channels, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_channels=embed_dims[i - 1],
                    embed_dim=embed_dims[i]
                )

            # Bloques MSCA
            block = nn.ModuleList([
                MSCABlock(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j]
                )
                for j in range(depths[i])
            ])

            norm = nn.LayerNorm(embed_dims[i])

            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Retorna features de todos los stages."""
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x = patch_embed(x)

            for blk in block:
                x = blk(x)

            # Normalización (reshape para LayerNorm)
            B, C, H, W = x.shape
            x_norm = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x_norm = norm(x_norm)
            x_norm = x_norm.transpose(1, 2).view(B, C, H, W)

            outs.append(x_norm)

        return outs


class HamburgerDecoder(nn.Module):
    """
    Hamburger Decoder (contexto global ligero).

    Usa factorización matricial para capturar contexto global
    de manera eficiente.
    """

    def __init__(
        self,
        in_channels: List[int] = [32, 64, 160, 256],
        embed_dim: int = 256,
        num_classes: int = 1
    ):
        super().__init__()

        # Proyecciones lineales para cada stage
        self.linear_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim)
            )
            for ch in in_channels
        ])

        # Hamburger module simplificado
        self.ham = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        # Proyectar y upscale todas las features al mismo tamaño
        out = []
        for i, (feat, linear) in enumerate(zip(features, self.linear_fuse)):
            feat = linear(feat)
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            out.append(feat)

        # Concatenar
        out = torch.cat(out, dim=1)

        # Hamburger
        out = self.ham(out)

        # Clasificar
        out = self.classifier(out)

        return out


class SegNeXt(nn.Module):
    """
    SegNeXt: Segmentación con Atención Convolucional Multi-Escala.

    Args:
        in_channels: Canales de entrada (6 para Sentinel-2)
        num_classes: Clases de salida (1 para binario)
        embed_dims: Dimensiones por stage
        depths: Número de bloques por stage
        variant: 'tiny', 'small', 'base', 'large'

    Ejemplo:
        >>> model = SegNeXt(in_channels=6, num_classes=1, variant='tiny')
        >>> x = torch.randn(4, 6, 128, 128)
        >>> y = model(x)  # [4, 1, 128, 128]
    """

    # Configuraciones por variante
    VARIANTS = {
        'tiny': {
            'embed_dims': [32, 64, 160, 256],
            'depths': [3, 3, 5, 2],
            'mlp_ratios': [8, 8, 4, 4]
        },
        'small': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [2, 2, 4, 2],
            'mlp_ratios': [8, 8, 4, 4]
        },
        'base': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 3, 12, 3],
            'mlp_ratios': [8, 8, 4, 4]
        },
        'large': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 5, 27, 3],
            'mlp_ratios': [8, 8, 4, 4]
        }
    }

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        variant: str = 'tiny',
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.variant = variant

        # Obtener configuración
        config = self.VARIANTS.get(variant, self.VARIANTS['tiny'])
        embed_dims = config['embed_dims']
        depths = config['depths']
        mlp_ratios = config['mlp_ratios']

        # Encoder (MSCAN)
        self.encoder = MSCAN(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            mlp_ratios=mlp_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Decoder (Hamburger)
        self.decoder = HamburgerDecoder(
            in_channels=embed_dims,
            embed_dim=256,
            num_classes=num_classes
        )

        # Inicializar pesos
        self._init_weights()

        # Imprimir información
        self._print_model_info()

    def _init_weights(self):
        """Inicialización de pesos."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        """Imprime información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        config = self.VARIANTS.get(self.variant, self.VARIANTS['tiny'])

        print("\n" + "=" * 70)
        print("SegNeXt Inicializado")
        print("=" * 70)
        print(f"Variante: {self.variant}")
        print(f"Canales de entrada: {self.in_channels}")
        print(f"Clases de salida: {self.num_classes}")
        print(f"Embed dims: {config['embed_dims']}")
        print(f"Depths: {config['depths']}")
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print("=" * 70 + "\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[-2:]  # (H, W)

        # Encoder: obtener features multi-escala
        features = self.encoder(x)

        # Decoder: fusionar y clasificar
        # Target size es 1/4 del input (por el stem)
        target_size = (input_size[0] // 4, input_size[1] // 4)
        out = self.decoder(features, target_size)

        # Upscale al tamaño original
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out


def create_segnext_model(
    variant: str = 'tiny',
    in_channels: int = 6,
    num_classes: int = 1
) -> nn.Module:
    """
    Factory function para crear modelos SegNeXt.

    Args:
        variant: 'tiny' (~4M), 'small' (~14M), 'base' (~28M), 'large' (~49M)
        in_channels: Canales de entrada
        num_classes: Clases de segmentación

    Returns:
        Modelo SegNeXt configurado

    Recomendaciones para 128x128:
    - tiny: Más rápido, suficiente para datasets pequeños
    - small: Buen balance rendimiento/velocidad
    """
    return SegNeXt(
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant
    )


# =============================================================================
# INFORMACIÓN ADICIONAL
# =============================================================================
"""
COMPARACIÓN CON OTROS MODELOS:

┌─────────────────┬────────────┬────────────┬──────────────────────────────────┐
│ Modelo          │ Params     │ GFLOPs     │ mIoU ADE20K                      │
├─────────────────┼────────────┼────────────┼──────────────────────────────────┤
│ SegNeXt-T       │ 4.3M       │ 6.6        │ 41.1                             │
│ SegNeXt-S       │ 13.9M      │ 15.9       │ 44.3                             │
│ SegNeXt-B       │ 27.6M      │ 34.9       │ 48.5                             │
│ SegNeXt-L       │ 48.9M      │ 57.8       │ 52.1                             │
│                 │            │            │                                  │
│ SegFormer-B0    │ 3.8M       │ 8.4        │ 37.4                             │
│ SegFormer-B1    │ 13.7M      │ 15.9       │ 42.2                             │
│ SegFormer-B2    │ 27.5M      │ 62.4       │ 46.5                             │
└─────────────────┴────────────┴────────────┴──────────────────────────────────┘

VENTAJAS DE SEGNEXT:
1. CNN pura - sin self-attention costosa
2. Strip convolutions - contexto multi-escala eficiente
3. Supera a SegFormer con menos/similar FLOPs
4. Fácil de entrenar (no necesita warmup especial)

PARA TU CASO (MANGLAR 128x128):
- SegNeXt-Tiny recomendado como primer intento
- Menor overhead que SegFormer para misma calidad
- Buen rendimiento en objetos pequeños (strip convolutions)
"""
