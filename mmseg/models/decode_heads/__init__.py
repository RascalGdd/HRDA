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
from .cffm_head_vpattn import CFFMHead_clips_resize1_8_vpattn
from .cffm_head_vpmove import CFFMHead_clips_resize1_8_vpmove
from .cffm_head_vpfuse import CFFMHead_clips_resize1_8_vpfuse
from .trans_scale_video_head import TransHeadVideo
from .daformer_head_video import DAFormerSerialHead
from .daformer_head_focal import DAFormerHeadFocal

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
    'CFFMHead_clips_resize1_8_vpattn',
    'CFFMHead_clips_resize1_8_vpmove',
    'CFFMHead_clips_resize1_8_vpfuse',
    'TransHeadVideo',
    'DAFormerSerialHead',
    'DAFormerHeadFocal'
]
