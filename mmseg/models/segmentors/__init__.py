# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add HRDAEncoderDecoder

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder, EncoderDecoder_clips
from .hrda_encoder_decoder import HRDAEncoderDecoder, HRDAEncoderDecoder_clips

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'EncoderDecoder_clips', 'HRDAEncoderDecoder_clips']
