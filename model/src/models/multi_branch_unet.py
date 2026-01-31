"""
Arquitectura Multi-Branch UNet++ para Sentinel-2 con Resoluciones Múltiples

Basado en:
- Cao et al. (2021): "Dual Stream Fusion Network for Multi-spectral High Resolution
  Remote Sensing Image Segmentation" (PRCV 2021)
- Liu et al. (2017): "Remote Sensing Image Fusion Based on Two-stream Fusion Network"
- Zhou et al. (2018): "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"

Esta arquitectura combina Multi-Branch con UNet++:
- High-Res Branch (10m): RGB + NIR (B2, B3, B4, B8)
- Low-Res Branch (20m): SWIR1 + SWIR2 (B11, B12)
- UNet++ Decoder: Nested skip connections y deep supervision

Características:
1. Procesamiento independiente de cada grupo de bandas
2. Upsampling de bandas de baja resolución
3. Fusión multi-escala con Feature Pyramid Network (FPN)
4. UNet++ decoder con nested skip connections para mejor flujo de gradientes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Optional


class MultiBranchUNet(nn.Module):
    """
    UNet++ Multi-Branch para procesamiento de bandas multi-resolución.

    Arquitectura:
    ```
    Input Sentinel-2:
    ├─ High-Res (10m): [B, C_high, H, W]  → Branch 1 (ResNet encoder)
    │                                      → Features @ multiple scales
    │                                      ↓
    └─ Low-Res (20m):  [B, C_low, H/2, W/2] → Upsample → Branch 2 (ResNet encoder)
                                            → Features @ multiple scales
                                            ↓
                                    [Fusion Module (FPN-style)]
                                            ↓
                                    [UNet++ Decoder]
                                    (Nested Skip Connections)
                                            ↓
                                        [Segmentation Map]
    ```

    Args:
        encoder_name: Nombre del encoder de SMP (e.g., 'resnet18', 'resnet50')
        encoder_weights: Pesos pre-entrenados ('imagenet', None, o custom)
        high_res_channels: Número de bandas de alta resolución (10m) - default: 4 (RGB+NIR)
        low_res_channels: Número de bandas de baja resolución (20m) - default: 2 (SWIR1+SWIR2)
        num_classes: Número de clases para segmentación (1 para binario)
        fusion_mode: Modo de fusión ('concat', 'add', 'fpn')
        upsample_mode: Modo de upsampling para bandas low-res ('bilinear', 'bicubic')
        deep_supervision: Habilitar deep supervision de UNet++ (default: False para compatibilidad)
        attention_type: Tipo de atención en decoder ('scse', None) - default: 'scse'
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = "imagenet",
        high_res_channels: int = 4,  # B2, B3, B4, B8
        low_res_channels: int = 2,   # B11, B12
        num_classes: int = 1,
        fusion_mode: str = "fpn",
        upsample_mode: str = "bilinear",
        deep_supervision: bool = False,
        attention_type: Optional[str] = "scse",
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.high_res_channels = high_res_channels
        self.low_res_channels = low_res_channels
        self.fusion_mode = fusion_mode
        self.upsample_mode = upsample_mode
        self.deep_supervision = deep_supervision
        self.attention_type = attention_type

        # ================================================================
        # Branch 1: High-Resolution Encoder (10m bands: RGB + NIR)
        # ================================================================
        self.high_res_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=high_res_channels,
            depth=5,
            weights=encoder_weights,
        )

        # ================================================================
        # Branch 2: Low-Resolution Encoder (20m bands: SWIR1 + SWIR2)
        # ================================================================
        self.low_res_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=low_res_channels,
            depth=5,
            weights=None,  # No hay pesos pre-entrenados para 2 canales
        )

        # ================================================================
        # Fusion Module
        # ================================================================
        # Obtener el número de canales de salida de cada nivel del encoder
        high_res_out_channels = self.high_res_encoder.out_channels
        low_res_out_channels = self.low_res_encoder.out_channels

        if fusion_mode == "fpn":
            # Feature Pyramid Network style fusion
            self.fusion_modules = nn.ModuleList([
                FusionBlock(
                    high_res_ch=high_res_out_channels[i],
                    low_res_ch=low_res_out_channels[i],
                    out_ch=high_res_out_channels[i],  # Mantener dimensiones del high-res
                )
                for i in range(len(high_res_out_channels))
            ])
            decoder_channels_in = high_res_out_channels
        elif fusion_mode == "concat":
            # Concatenación simple
            decoder_channels_in = [
                h + l for h, l in zip(high_res_out_channels, low_res_out_channels)
            ]
        elif fusion_mode == "add":
            # Suma (requiere same channels)
            assert high_res_out_channels == low_res_out_channels, \
                "Para fusion_mode='add', los encoders deben tener mismo número de canales"
            decoder_channels_in = high_res_out_channels
        else:
            raise ValueError(f"fusion_mode '{fusion_mode}' no soportado")

        # ================================================================
        # UNet++ Decoder (Nested U-Net)
        # ================================================================
        # UNet++ usa un decoder más complejo con nested skip connections
        self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=decoder_channels_in,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_norm="batchnorm",  # Cambiado de use_batchnorm a use_norm
            center=False,
            attention_type=self.attention_type,
            interpolation_mode="nearest",
        )

        # ================================================================
        # Segmentation Head
        # ================================================================
        if deep_supervision:
            # Deep supervision: múltiples outputs en diferentes niveles
            # Útil durante entrenamiento, se puede desactivar en inferencia
            self.segmentation_heads = nn.ModuleList([
                smp.base.SegmentationHead(
                    in_channels=ch,
                    out_channels=num_classes,
                    activation=None,
                    kernel_size=3,
                )
                for ch in [16, 32, 64, 128]  # Diferentes niveles del decoder
            ])
        else:
            # Single output (más común en producción)
            self.segmentation_head = smp.base.SegmentationHead(
                in_channels=16,
                out_channels=num_classes,
                activation=None,
                kernel_size=3,
            )

        print(f"\n{'='*70}")
        print(f"Multi-Branch UNet++ Inicializado")
        print(f"{'='*70}")
        print(f"Encoder: {encoder_name}")
        print(f"Decoder: UNet++ (Nested Skip Connections)")
        print(f"Attention: {attention_type if attention_type else 'None'}")
        print(f"High-Res Branch: {high_res_channels} canales (10m resolution)")
        print(f"Low-Res Branch: {low_res_channels} canales (20m resolution)")
        print(f"Fusion Mode: {fusion_mode}")
        print(f"Upsampling: {upsample_mode}")
        print(f"Deep Supervision: {deep_supervision}")
        print(f"Clases: {num_classes}")
        print(f"{'='*70}\n")

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del Multi-Branch UNet++.

        Args:
            x_high: Tensor de alta resolución [B, C_high, H, W]
            x_low: Tensor de baja resolución [B, C_low, H/2, W/2]

        Returns:
            Si deep_supervision=False:
                Mapa de segmentación [B, num_classes, H, W]
            Si deep_supervision=True:
                Lista de mapas de segmentación en diferentes escalas
        """

        # ================================================================
        # 1. Upsample Low-Res Input to match High-Res spatial dimensions
        # ================================================================
        x_low_upsampled = F.interpolate(
            x_low,
            size=x_high.shape[-2:],  # (H, W)
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode != 'nearest' else None
        )

        # ================================================================
        # 2. Encoding: Extract multi-scale features from both branches
        # ================================================================
        high_res_features = self.high_res_encoder(x_high)
        low_res_features = self.low_res_encoder(x_low_upsampled)

        # ================================================================
        # 3. Fusion: Combine features from both branches at each scale
        # ================================================================
        if self.fusion_mode == "fpn":
            fused_features = [
                self.fusion_modules[i](high_res_features[i], low_res_features[i])
                for i in range(len(high_res_features))
            ]
        elif self.fusion_mode == "concat":
            fused_features = [
                torch.cat([high_res_features[i], low_res_features[i]], dim=1)
                for i in range(len(high_res_features))
            ]
        elif self.fusion_mode == "add":
            fused_features = [
                high_res_features[i] + low_res_features[i]
                for i in range(len(high_res_features))
            ]

        # ================================================================
        # 4. Decoding: UNet++ Nested Skip Connections
        # ================================================================
        # UNet++ decoder espera recibir una lista de features como un solo argumento
        decoder_outputs = self.decoder(fused_features)

        # ================================================================
        # 5. Segmentation Head: Generate predictions
        # ================================================================
        if self.deep_supervision:
            # Deep supervision: retornar múltiples outputs
            # Útil para entrenamiento con supervisión auxiliar
            segmentation_maps = []
            for i, output in enumerate(decoder_outputs):
                seg_map = self.segmentation_heads[i](output)
                # Upsample to original resolution
                seg_map = F.interpolate(
                    seg_map,
                    size=x_high.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                segmentation_maps.append(seg_map)
            return segmentation_maps  # Lista de mapas en diferentes escalas
        else:
            # Single output (modo inferencia/producción)
            # Usar solo el output más fino (primer elemento)
            segmentation_map = self.segmentation_head(decoder_outputs)
            return segmentation_map


class FusionBlock(nn.Module):
    """
    Bloque de fusión tipo FPN para combinar características de dos ramas.

    Implementa:
    1. Proyección de características a espacio común
    2. Fusión mediante suma ponderada o concatenación
    3. Refinamiento con convoluciones
    """

    def __init__(self, high_res_ch: int, low_res_ch: int, out_ch: int):
        super().__init__()

        # Proyectar high-res features
        self.high_res_proj = nn.Sequential(
            nn.Conv2d(high_res_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Proyectar low-res features
        self.low_res_proj = nn.Sequential(
            nn.Conv2d(low_res_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Refinamiento post-fusión
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_res_feat: torch.Tensor, low_res_feat: torch.Tensor) -> torch.Tensor:
        """
        Fusiona características de alta y baja resolución.

        Args:
            high_res_feat: Features de alta resolución [B, C_h, H, W]
            low_res_feat: Features de baja resolución [B, C_l, H, W]

        Returns:
            Features fusionadas [B, out_ch, H, W]
        """
        # Proyectar a espacio común
        high_proj = self.high_res_proj(high_res_feat)
        low_proj = self.low_res_proj(low_res_feat)

        # Fusión aditiva
        fused = high_proj + low_proj

        # Refinamiento
        refined = self.refine(fused)

        return refined


# ============================================================================
# Wrapper para Compatibilidad con TorchGeo DataModule
# ============================================================================

class MultiBranchUNetWrapper(nn.Module):
    """
    Wrapper que separa automáticamente las bandas high-res y low-res
    del input tensor y las pasa al Multi-Branch UNet++.

    Esto permite usar el modelo directamente con TorchGeo sin cambios
    en el DataModule.

    Asume que el orden de bandas en el input es:
    - Canales 0-3: High-res (B2, B3, B4, B8)
    - Canales 4-5: Low-res (B11, B12)
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = "imagenet",
        high_res_channels: int = 4,
        low_res_channels: int = 2,
        num_classes: int = 1,
        fusion_mode: str = "fpn",
        upsample_mode: str = "bilinear",
        deep_supervision: bool = False,
        attention_type: Optional[str] = "scse",
    ):
        super().__init__()

        self.high_res_channels = high_res_channels
        self.low_res_channels = low_res_channels
        self.deep_supervision = deep_supervision

        self.model = MultiBranchUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            high_res_channels=high_res_channels,
            low_res_channels=low_res_channels,
            num_classes=num_classes,
            fusion_mode=fusion_mode,
            upsample_mode=upsample_mode,
            deep_supervision=deep_supervision,
            attention_type=attention_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con separación automática de bandas.

        Args:
            x: Tensor de entrada [B, C_total, H, W] donde C_total = high_res_channels + low_res_channels

        Returns:
            Mapa de segmentación [B, num_classes, H, W]
        """
        # Separar bandas high-res y low-res
        x_high = x[:, :self.high_res_channels, :, :]  # [B, 4, H, W]
        x_low = x[:, self.high_res_channels:, :, :]   # [B, 2, H, W]

        # Forward a través del Multi-Branch U-Net
        return self.model(x_high, x_low)
