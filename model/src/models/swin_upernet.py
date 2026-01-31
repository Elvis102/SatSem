"""
Swin Transformer V2 + UPerNet para Segmentación Semántica.

Swin Transformer V2 es una mejora sobre Swin V1 con:
- Log-spaced Continuous Position Bias (mejor escalabilidad)
- Scaled cosine attention (más estable)
- Mejor transferencia entre resoluciones

UPerNet (Unified Perceptual Parsing Network) es un framework de segmentación
que combina:
- Feature Pyramid Network (FPN) para multi-escala
- Pyramid Pooling Module (PPM) para contexto global
- Compatible con cualquier backbone visual

Referencias:
- Liu et al. (2022): "Swin Transformer V2: Scaling Up Capacity and Resolution"
- Xiao et al. (2018): "Unified Perceptual Parsing for Scene Understanding"
- https://huggingface.co/docs/transformers/model_doc/swinv2
- https://huggingface.co/docs/transformers/model_doc/upernet

RENDIMIENTO REPORTADO (ADE20K):
- Swin-T + UPerNet: 44.5 mIoU
- Swin-S + UPerNet: 47.6 mIoU
- Swin-B + UPerNet: 48.1 mIoU
- Swin-L + UPerNet: 53.5 mIoU (ImageNet-22K pretrained)

MEJORAS DE SWIN V2 SOBRE V1:
┌────────────────────────────────────────────────────────────────────────┐
│              Swin V1                 │            Swin V2              │
├──────────────────────────────────────┼─────────────────────────────────┤
│ Relative position bias (tabla)       │ Log-spaced continuous bias      │
│ Dot-product attention                │ Scaled cosine attention         │
│ Difícil escalar a alta resolución    │ Mejor transferencia resolución  │
│ Hasta 1.5B params estable            │ Hasta 3B params estable         │
└──────────────────────────────────────┴─────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Swinv2Config,
    Swinv2Model,
    UperNetConfig,
    UperNetForSemanticSegmentation,
    AutoBackbone
)
from typing import Optional, List
import warnings


class SwinV2UPerNetWrapper(nn.Module):
    """
    Wrapper de Swin Transformer V2 + UPerNet para segmentación semántica.

    Adapta el modelo para:
    1. Aceptar 6 canales de entrada (Sentinel-2)
    2. Producir segmentación binaria
    3. Interpolar salida al tamaño de entrada

    Args:
        model_size: Tamaño del modelo ('tiny', 'small', 'base', 'large')
        in_channels: Canales de entrada (6 para Sentinel-2)
        num_classes: Clases de salida (1 para binario)
        pretrained: Usar pesos preentrenados
        window_size: Tamaño de ventana para attention (default: 7 para 128x128)

    Ejemplo:
        >>> model = SwinV2UPerNetWrapper(
        ...     model_size='tiny',
        ...     in_channels=6,
        ...     num_classes=1
        ... )
        >>> x = torch.randn(4, 6, 128, 128)
        >>> y = model(x)  # [4, 1, 128, 128]
    """

    # Configuraciones por tamaño
    MODEL_CONFIGS = {
        'tiny': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'small': {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'base': {
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7,
        },
        'large': {
            'embed_dim': 192,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'window_size': 7,
        }
    }

    def __init__(
        self,
        model_size: str = 'tiny',
        in_channels: int = 6,
        num_classes: int = 1,
        pretrained: bool = True,
        window_size: int = 7,
        image_size: int = 128
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model_size = model_size

        # Para segmentación binaria, usamos 2 clases internamente (background + foreground)
        # y luego convertimos a logits binarios en forward
        self.internal_num_classes = 2 if num_classes == 1 else num_classes

        # Obtener configuración
        config_params = self.MODEL_CONFIGS.get(model_size, self.MODEL_CONFIGS['tiny'])

        # Crear configuración de Swin V2
        backbone_config = Swinv2Config(
            image_size=image_size,
            patch_size=4,
            num_channels=3,  # Se modificará después
            embed_dim=config_params['embed_dim'],
            depths=config_params['depths'],
            num_heads=config_params['num_heads'],
            window_size=window_size,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )

        # Crear configuración de UPerNet
        upernet_config = UperNetConfig(
            backbone_config=backbone_config,
            num_labels=self.internal_num_classes,  # Usar 2 clases para binario
            use_pretrained_backbone=False,
        )

        # Crear modelo
        self.model = UperNetForSemanticSegmentation(upernet_config)

        # Adaptar para 6 canales
        self._adapt_input_channels(in_channels)

        # Nota: No cargamos pesos preentrenados de Swin V1 en Swin V2
        # ya que las arquitecturas son diferentes
        if pretrained:
            print("⚠️  Nota: Pesos preentrenados de UPerNet-Swin V1 no son compatibles con Swin V2")
            print("   El modelo se entrenará desde cero con inicialización por defecto.")

        # Imprimir información
        self._print_model_info()

    def _adapt_input_channels(self, in_channels: int):
        """Adapta la primera capa para 6 canales de entrada."""
        # Acceder al patch embedding del backbone
        backbone = self.model.backbone

        if hasattr(backbone, 'embeddings'):
            patch_embed = backbone.embeddings.patch_embeddings
            if hasattr(patch_embed, 'projection'):
                first_conv = patch_embed.projection

                if first_conv.in_channels == in_channels:
                    return

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
                    rgb_mean = original_weight.mean(dim=1, keepdim=True)

                    new_weight = torch.zeros(
                        first_conv.out_channels,
                        in_channels,
                        *first_conv.kernel_size
                    )

                    # Copiar RGB
                    new_weight[:, :3, :, :] = original_weight
                    # Extender
                    for i in range(3, in_channels):
                        new_weight[:, i:i+1, :, :] = rgb_mean

                    new_weight *= 3.0 / in_channels
                    new_conv.weight.data = new_weight

                    if first_conv.bias is not None:
                        new_conv.bias.data = first_conv.bias.data.clone()

                patch_embed.projection = new_conv

    def _load_pretrained_weights(self, model_size: str):
        """Intenta cargar pesos preentrenados de HuggingFace."""
        # Mapeo a modelos preentrenados de OpenMMLab
        pretrained_models = {
            'tiny': 'openmmlab/upernet-swin-tiny',
            'small': 'openmmlab/upernet-swin-small',
            'base': 'openmmlab/upernet-swin-base',
            'large': 'openmmlab/upernet-swin-large',
        }

        model_name = pretrained_models.get(model_size)
        if model_name:
            try:
                # Cargar modelo preentrenado
                pretrained = UperNetForSemanticSegmentation.from_pretrained(
                    model_name,
                    ignore_mismatched_sizes=True
                )

                # Copiar pesos del backbone (excepto primera capa)
                state_dict = pretrained.state_dict()
                own_state = self.model.state_dict()

                for name, param in state_dict.items():
                    if name in own_state:
                        if own_state[name].shape == param.shape:
                            own_state[name].copy_(param)
                        else:
                            # Skip mismatched layers (primera capa, classifier)
                            pass

                print(f"Pesos parcialmente cargados de: {model_name}")
            except Exception as e:
                warnings.warn(f"No se pudieron cargar pesos preentrenados: {e}")

    def _print_model_info(self):
        """Imprime información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        config = self.MODEL_CONFIGS.get(self.model_size, self.MODEL_CONFIGS['tiny'])

        print("\n" + "=" * 70)
        print("Swin Transformer V2 + UPerNet Inicializado")
        print("=" * 70)
        print(f"Tamaño: {self.model_size}")
        print(f"Canales de entrada: {self.in_channels}")
        print(f"Clases de salida: {self.num_classes} (interno: {self.internal_num_classes})")
        print(f"Embed dim: {config['embed_dim']}")
        print(f"Depths: {config['depths']}")
        print(f"Num heads: {config['num_heads']}")
        print(f"Window size: {config['window_size']}")
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print("=" * 70 + "\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[-2:]

        # Forward de UPerNet
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, internal_num_classes, H', W']

        # Interpolar al tamaño original
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )

        # Para segmentación binaria, convertir de 2 clases a 1 logit
        # logits = foreground - background (como log-odds ratio)
        if self.num_classes == 1 and logits.shape[1] == 2:
            logits = logits[:, 1:2, :, :] - logits[:, 0:1, :, :]

        return logits


class SwinV2UNet(nn.Module):
    """
    Swin Transformer V2 como encoder con decoder tipo UNet.

    Alternativa más ligera a UPerNet, usando skip connections
    estilo UNet para mejor preservación de detalles.

    Args:
        model_size: 'tiny', 'small', 'base', 'large'
        in_channels: Canales de entrada
        num_classes: Clases de salida
    """

    MODEL_CONFIGS = {
        'tiny': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
        'small': {'embed_dim': 96, 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
        'base': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]},
        'large': {'embed_dim': 192, 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48]},
    }

    def __init__(
        self,
        model_size: str = 'tiny',
        in_channels: int = 6,
        num_classes: int = 1,
        image_size: int = 128
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model_size = model_size

        # Para segmentación binaria, usamos 2 clases internamente
        self.internal_num_classes = 2 if num_classes == 1 else num_classes

        config_params = self.MODEL_CONFIGS.get(model_size, self.MODEL_CONFIGS['tiny'])
        embed_dim = config_params['embed_dim']

        # Encoder: Swin V2
        swin_config = Swinv2Config(
            image_size=image_size,
            patch_size=4,
            num_channels=in_channels,
            embed_dim=embed_dim,
            depths=config_params['depths'],
            num_heads=config_params['num_heads'],
            window_size=7,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )

        self.encoder = Swinv2Model(swin_config)

        # Calcular canales por stage
        # Stage 1: embed_dim, Stage 2: 2*embed_dim, Stage 3: 4*embed_dim, Stage 4: 8*embed_dim
        encoder_channels = [embed_dim * (2 ** i) for i in range(4)]

        # Decoder: Bloques de upsampling con skip connections
        self.decoder4 = self._make_decoder_block(encoder_channels[3], encoder_channels[2])
        self.decoder3 = self._make_decoder_block(encoder_channels[2] * 2, encoder_channels[1])
        self.decoder2 = self._make_decoder_block(encoder_channels[1] * 2, encoder_channels[0])
        self.decoder1 = self._make_decoder_block(encoder_channels[0] * 2, 64)

        # Upsampling final y clasificador
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Usar internal_num_classes para el clasificador
        self.classifier = nn.Conv2d(32, self.internal_num_classes, kernel_size=1)

        self._init_weights()
        self._print_model_info()

    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Crea un bloque decoder con upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        """Inicialización de pesos."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        """Imprime información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        print("\n" + "=" * 70)
        print("Swin V2 UNet Inicializado")
        print("=" * 70)
        print(f"Tamaño: {self.model_size}")
        print(f"Canales entrada: {self.in_channels}")
        print(f"Clases salida: {self.num_classes} (interno: {self.internal_num_classes})")
        print(f"Parámetros totales: {total_params:,}")
        print("=" * 70 + "\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con skip connections."""
        input_size = x.shape[-2:]

        # Encoder
        outputs = self.encoder(x, output_hidden_states=True)

        # Usar reshaped_hidden_states si está disponible (ya en formato [B, C, H, W])
        # Estructura: [stage1_out, stage2_out, stage3_out, stage4_out, final_out]
        # Tamaños para 128x128: [32x32, 16x16, 8x8, 4x4, 4x4]
        if hasattr(outputs, 'reshaped_hidden_states') and outputs.reshaped_hidden_states is not None:
            features = list(outputs.reshaped_hidden_states[:4])  # Solo stages 1-4
        else:
            # Fallback: reshape manual de hidden_states
            features = []
            for hs in outputs.hidden_states[:4]:
                B, L, C = hs.shape
                H = W = int(L ** 0.5)
                feat = hs.transpose(1, 2).view(B, C, H, W)
                features.append(feat)

        # features[0]: 32x32, features[1]: 16x16, features[2]: 8x8, features[3]: 4x4
        # Decoder con skip connections (de mayor a menor resolución)
        x = self.decoder4(features[3])  # 4x4 -> 8x8
        x = torch.cat([x, features[2]], dim=1)  # concat con 8x8

        x = self.decoder3(x)  # 8x8 -> 16x16
        x = torch.cat([x, features[1]], dim=1)  # concat con 16x16

        x = self.decoder2(x)  # 16x16 -> 32x32
        x = torch.cat([x, features[0]], dim=1)  # concat con 32x32

        x = self.decoder1(x)  # 32x32 -> 64x64 (parcialmente upsampled)

        # Final upsampling y clasificación
        x = self.final_up(x)  # 64x64 -> 128x128 (o tamaño original)
        logits = self.classifier(x)

        # Asegurar tamaño correcto
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        # Para segmentación binaria, convertir de 2 clases a 1 logit
        if self.num_classes == 1 and logits.shape[1] == 2:
            logits = logits[:, 1:2, :, :] - logits[:, 0:1, :, :]

        return logits


def create_swinv2_model(
    variant: str = 'tiny',
    decoder: str = 'upernet',
    in_channels: int = 6,
    num_classes: int = 1,
    image_size: int = 128
) -> nn.Module:
    """
    Factory function para crear modelos Swin V2.

    Args:
        variant: 'tiny' (~28M), 'small' (~50M), 'base' (~88M), 'large' (~197M)
        decoder: 'upernet' (mejor calidad) o 'unet' (más ligero)
        in_channels: Canales de entrada
        num_classes: Clases de segmentación
        image_size: Tamaño de imagen de entrada

    Returns:
        Modelo Swin V2 configurado

    Recomendaciones para 128x128:
    ┌────────────────┬─────────────────────────────────────────────┐
    │ Variante       │ Recomendación                               │
    ├────────────────┼─────────────────────────────────────────────┤
    │ tiny + unet    │ Más rápido, bueno para experimentos         │
    │ tiny + upernet │ Balance calidad/velocidad                   │
    │ small + upernet│ Mayor capacidad si hay suficientes datos    │
    └────────────────┴─────────────────────────────────────────────┘
    """
    if decoder == 'unet':
        return SwinV2UNet(
            model_size=variant,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size
        )
    else:  # upernet
        return SwinV2UPerNetWrapper(
            model_size=variant,
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size
        )


# =============================================================================
# INFORMACIÓN ADICIONAL
# =============================================================================
"""
COMPARACIÓN SWIN V1 vs V2:

Swin V2 introduce mejoras clave para escalar a modelos más grandes:

1. LOG-SPACED CONTINUOUS POSITION BIAS:
   - V1: Tabla de bias posicionales (tamaño fijo)
   - V2: Bias continuo con función log-espaciada
   - Beneficio: Mejor transferencia entre diferentes resoluciones

2. SCALED COSINE ATTENTION:
   - V1: Dot-product attention estándar
   - V2: Cosine attention con factor de escala learnable
   - Beneficio: Entrenamiento más estable con modelos grandes

3. RENDIMIENTO:
   - ImageNet-1K: 84.0% (V1) vs 84.3% (V2) con modelo tiny
   - ADE20K: +0.5-1.0 mIoU con misma arquitectura

COMPARACIÓN CON OTROS MODELOS PARA MANGLARES:

┌─────────────────────┬────────────┬──────────────────────────────────────┐
│ Modelo              │ Params     │ Características                      │
├─────────────────────┼────────────┼──────────────────────────────────────┤
│ Swin V2-T + UPerNet │ ~28M       │ Shifted windows, contexto jerárquico │
│ SegNeXt-Tiny        │ ~4M        │ CNN con atención multi-escala        │
│ SegFormer-B0        │ ~3.7M      │ Transformer eficiente                │
│ Mask2Former-Tiny    │ ~47M       │ SOTA universal                       │
└─────────────────────┴────────────┴──────────────────────────────────────┘

CUÁNDO USAR SWIN V2:
- Cuando se necesita balance entre Transformer y eficiencia
- Para transferencia desde pesos ImageNet
- Cuando se tiene GPU con >8GB VRAM
"""
