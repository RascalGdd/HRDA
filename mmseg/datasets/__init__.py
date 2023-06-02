# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .acdc import ACDCDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .cityscapes_depth import CityscapesDatasetDepth
from .custom import CustomDataset
from .custom_depth import CustomDatasetDepth
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .gta import GTADataset
from .synthia import SynthiaDataset
from .uda_dataset import UDADataset

__all__ = [
    'CustomDataset',
    'CustomDatasetDepth',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'CityscapesDatasetDepth'
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
]
