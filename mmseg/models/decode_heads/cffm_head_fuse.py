import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips_flow
from mmseg.models.utils import *
import attr

from IPython import embed
from .cffm_module.cffm_transformer import BasicLayer3d3

import cv2

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], embed_dim=768):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

    def forward(self, inputs):
    # inputs: [c1, c2, c3, c4]
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        return _c

@HEADS.register_module()
class CFFMHeadFuse(BaseDecodeHead_clips_flow):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(CFFMHeadFuse, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.seg_fuse = FeatureFusion(self.in_channels, embedding_dim)
        self.attn_fuse = FeatureFusion(self.in_channels, embedding_dim)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.attn_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)
        # self.simple_seg_fuse = nn.Conv2d(self.num_classes*2, self.num_classes, kernel_size=1)

        depths = decoder_params['depths']

        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               input_resolution=(60,
                                 60),
               depth=depths,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2, 
               focal_window=5, 
               expand_size=3, 
               expand_layer="all",                           
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               use_layerscale=False, 
               layerscale_value=1e-4,
               focal_l_clips=[1,2,3],
               focal_kernel_clips=[7,5,3])

        print(self.decoder_focal.blocks[0].focal_kernel_clips)

    def forward(self, inputs):
        # if self.training:
        #     assert self.num_clips==num_clips
        num_clips = self.num_clips

        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        n, _, _, _ = c4.shape
        batch_size = int(n / num_clips)
        x_single = [c.reshape(batch_size, num_clips, -1, c.shape[2], c.shape[3])[:,-1].detach() for c in x]

        _c = self.seg_fuse(x)
        _att = self.attn_fuse(x_single)

        _, _, h, w = _c.shape
        x = self.dropout(_c) # Bk=4, C, H, W
        default_logit = self.linear_pred(x.reshape(batch_size, num_clips, -1, x.shape[2], x.shape[3])[:,-1]) # 1, C, H2, W2
        att = torch.sigmoid(self.attn_pred(_att))

        h2=int(h/2)
        w2=int(w/2)
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)

        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)

        _c2=self.decoder_focal(_c_further)
        assert _c_further.shape==_c2.shape

        _c_further2=torch.cat([_c_further[:,-1], _c2[:,-1]],1) #  B, 2*Embdim, H2, W2

        x2 = self.dropout(_c_further2)
        x2 = self.linear_pred2(x2)
        focal_logit = resize(x2, size=(h,w),mode='bilinear',align_corners=False) # B, C, H, W

        # fused_logit = self.simple_seg_fuse(torch.cat([default_logit, focal_logit], dim=1)) # B, C, H, W
        fused_logit = att * focal_logit + (1-att) * default_logit

        return fused_logit

