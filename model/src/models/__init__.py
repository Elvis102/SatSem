# Arquitecturas de modelos para segmentación multi-resolución
"""
Modelos disponibles para segmentación de manglares con Sentinel-2:

CNN CLÁSICAS:
1. MultiBranchUNetWrapper: UNet++ multi-resolución (10m + 20m)
2. SegNet/SegNetLite: Arquitectura clásica encoder-decoder

TRANSFORMERS:
3. SegFormerWrapper: Vision Transformer eficiente
4. Mask2FormerWrapper: SOTA con masked attention
5. SwinV2UPerNetWrapper: Swin Transformer V2 + UPerNet (SOTA jerárquico)

CNN MODERNAS:
6. SegNeXt: CNN con atención convolucional multi-escala (SOTA eficiente)

Uso:
    from src.models import MultiBranchUNetWrapper, SegFormerWrapper
    from src.models import SegNet, SegNeXt, Mask2FormerWrapper
    from src.models import SwinV2UPerNetWrapper, create_swinv2_model
"""

from .multi_branch_unet import MultiBranchUNetWrapper
from .segformer_wrapper import (
    SegFormerWrapper,
    SegFormerWithAuxLoss,
    create_segformer_model
)
from .segnet import (
    SegNet,
    SegNetLite,
    create_segnet_model
)
from .segnext import (
    SegNeXt,
    create_segnext_model
)
from .mask2former_wrapper import (
    Mask2FormerWrapper,
    create_mask2former_model
)
from .swin_upernet import (
    SwinV2UPerNetWrapper,
    SwinV2UNet,
    create_swinv2_model
)

__all__ = [
    # CNN clásicas
    'MultiBranchUNetWrapper',
    'SegNet',
    'SegNetLite',
    'create_segnet_model',
    # Transformers
    'SegFormerWrapper',
    'SegFormerWithAuxLoss',
    'create_segformer_model',
    'Mask2FormerWrapper',
    'create_mask2former_model',
    # CNN modernas
    'SegNeXt',
    'create_segnext_model',
    # Swin Transformer V2
    'SwinV2UPerNetWrapper',
    'SwinV2UNet',
    'create_swinv2_model',
]
