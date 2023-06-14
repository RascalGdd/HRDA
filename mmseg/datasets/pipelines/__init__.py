# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .compose import Compose
from .formating import (Collect, ImageToTensor, ImageToTensor_clips, ToDataContainer, ToTensor,
                        Transpose, DefaultFormatBundle,  DefaultFormatBundle_clips, to_tensor)
from .loading import LoadAnnotations, LoadImageFromFile
from .test_time_aug import MultiScaleFlipAug
from .transforms import (CLAHE, AdjustGamma, Normalize, Pad,
                         PhotoMetricDistortion, RandomCrop, RandomFlip,
                         RandomRotate, Rerange, Resize, RGB2Gray, SegRescale)
from .transforms_clips import (AlignedResize_clips, RandomFlip_clips,
                                Pad_clips, Normalize_clips, RandomCrop_clips,
                                PhotoMetricDistortion_clips)

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'LoadAnnotations', 'LoadImageFromFile',
    'MultiScaleFlipAug', 'Resize', 'RandomFlip', 'Pad', 'RandomCrop',
    'Normalize', 'SegRescale', 'PhotoMetricDistortion', 'RandomRotate',
    'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
    'AlignedResize_clips', 'RandomFlip_clips',
    'Pad_clips', 'Normalize_clips', 'RandomCrop_clips',
    'PhotoMetricDistortion_clips'
]
