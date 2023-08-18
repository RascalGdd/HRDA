# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .hrda_head import HRDAHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead

from .cffm_head import CFFMHead_clips_resize1_8
from .cffm_head_vp import CFFMHead_clips_resize1_8_vp
from .cffm_head_fuse import CFFMHeadFuse
from .trans_scale_video_head import TransHeadVideo
from .daformer_head_video import DAFormerSerialHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
    'HRDAHead',
    'CFFMHead_clips_resize1_8',
    'CFFMHead_clips_resize1_8_vp',
    'CFFMHeadFuse',
    'TransHeadVideo',
    'DAFormerSerialHead'
]
