# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips_flow

from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2

def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


CFFM_head_config_b0 = dict(
    type='CFFMHead_clips_resize1_8',
    in_channels=[32, 64, 160, 256],
    in_index=[0, 1, 2, 3],
    feature_strides=[4, 8, 16, 32],
    channels=128,
    dropout_ratio=0.1,
    num_classes=19,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    decoder_params=dict(embed_dim=256, depths=2),
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
)

CFFM_head_config_b1 = dict(
    type='CFFMHead_clips_resize1_8',
    in_channels=[32, 64, 160, 256],
    in_index=[0, 1, 2, 3],
    feature_strides=[4, 8, 16, 32],
    channels=128,
    dropout_ratio=0.1,
    num_classes=19,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    decoder_params=dict(embed_dim=256, depths=2),
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
)

CFFM_head_config_b3 = dict(
    type='CFFMHead_clips_resize1_8',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    feature_strides=[4, 8, 16, 32],
    channels=256,
    dropout_ratio=0.1,
    num_classes=19,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    decoder_params=dict(embed_dim=256, depths=2),
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
)

daformer_focal_head_config_b3 = dict(
    type='DAFormerHeadFocal',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    feature_strides=[4, 8, 16, 32],
    channels=256,
    dropout_ratio=0.1,
    num_classes=19,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    decoder_params=dict(
        embed_dims=256,
        embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
        embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
        fusion_cfg=dict(
            type='conv',
            kernel_size=1,
            act_cfg=dict(type='ReLU'),
            norm_cfg=norm_cfg),
        depths=2,
        cffm_downsample=False
    ),
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
)


@HEADS.register_module()
class HRDAHead(BaseDecodeHead_clips_flow):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)

        attn_cfg['channels'] = attention_embed_dim
        attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
        if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
            attn_cfg['decoder_params']['fusion_cfg'] = dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                ['norm_cfg'])
        kwargs['init_cfg'] = None
        kwargs['input_transform'] = 'multiple_select'
        self.os = 4
        if 'num_clips' in attn_cfg:
            attn_cfg.pop('num_clips')

        if single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        elif 'CFFM' in single_scale_head: # only for video clips
            self.num_clips = kwargs['num_clips']

            if 'b0' in single_scale_head:
                head_cfg = CFFM_head_config_b0
            elif 'b1' in single_scale_head:
                head_cfg = CFFM_head_config_b1
            elif 'b3' in single_scale_head:
                head_cfg = CFFM_head_config_b3
            else:
                assert 0, "specify the backbone in CFFMHead, e.g., CFFMHead_b0"

            if 'vpattn' in single_scale_head:
                head_cfg['type'] = head_cfg['type'] + '_vpattn'

            if 'vpmove' in single_scale_head:
                head_cfg['type'] = head_cfg['type'] + '_vpmove'

            if 'vpfuse' in single_scale_head:
                head_cfg['type'] = head_cfg['type'] + '_vpfuse'

            head_cfg["num_clips"] = self.num_clips

        elif 'DAFormerFocal' in single_scale_head:
            self.num_clips = kwargs['num_clips']
            if 'b3' in single_scale_head:
                head_cfg = daformer_focal_head_config_b3
            else:
                assert 0, "specify the backbone in CFFMHead, e.g., DAFormerFocal_b3, only b3 is supported"
            head_cfg["num_clips"] = self.num_clips
            if 'down' in single_scale_head:
                head_cfg['decoder_params']['cffm_downsample'] = True

        elif single_scale_head == 'DAFormerHead':
            head_cfg['type'] = single_scale_head
            if 'num_clips' in head_cfg:
                head_cfg.pop('num_clips')
        else:
            raise NotImplementedError(single_scale_head)

        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        self.single_scale_head = single_scale_head
        self.head_type = head_cfg['type']
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = 'DAFormerHead'
        if 'TransHead' in single_scale_head:
            attn_cfg['type'] = 'TransHeadVideo'
        if 'serial' in single_scale_head:
            attn_cfg['type'] = 'DAFormerSerialHead'

        if 'CFFMLR' in single_scale_head:
            self.cffm_only_lr = True
        else:
            self.cffm_only_lr = False 

        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference

        self.debug_output_attention = debug_output_attention
        self.debug = False
        self.debug_cnt = 0

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device

            # debug
            # print("decode hr input (testing)", features[0][0].shape)

            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            if self.cffm_only_lr:
                crop_seg_logits = self.head(features, no_cffm = True)
            else:
                crop_seg_logits = self.head(features)
                
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            # debug
            # print("decode hr input (training)", inp[0].shape)
            return self.head(inp)

    def get_scale_attention(self, inp, feat_video = None, lr_out = None, hr_out = None):

        # TODO: underlying assumption num_per_gpu = 1, and the last clip is the current clip
        if inp[0].shape[0] == self.num_clips:
            for i in range(len(inp)):
                inp[i] = inp[i][-1:]

        if self.scale_attention is not None:
            if feat_video is None:
                att = torch.sigmoid(self.scale_attention(inp))
            else:
                if lr_out is not None and hr_out is not None and 'serial' in self.head_type:
                    att = torch.sigmoid(self.scale_attention(inp, feat_video, lr_out, hr_out))
                else:
                    att = torch.sigmoid(self.scale_attention(inp, feat_video = feat_video))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2

        batch_size = int(inputs[0][0].shape[0] / self.num_clips)
        
        # convert video feature to single image feature for HRDA-only mode
        if 'CFFM' not in self.single_scale_head:
            new_inputs = [[], []]
            for i_level in range(2):
                if type(inputs[i_level]) == list:
                    for i in range(len(inputs[i_level])):
                        new_inputs[i_level].append(
                            inputs[i_level][i].reshape(
                                batch_size, self.num_clips, -1, inputs[i_level][i].shape[2], inputs[i_level][i].shape[3]
                            )[:,-1]
                        )
                elif type(inputs[i_level]) == dict:
                    new_inputs[i_level] = {
                        'features': [], 'boxes': inputs[i_level]['boxes']
                    }
                    hr_batch_size = int(inputs[i_level]['features'][0].shape[0] / self.num_clips)
                    for i in range(len(inputs[i_level]['features'])):
                        new_inputs[i_level]['features'].append(
                            inputs[i_level]['features'][i].reshape(
                                hr_batch_size, self.num_clips, -1, inputs[i_level]['features'][i].shape[2], inputs[i_level]['features'][i].shape[3]
                            )[:,-1]
                        )

            hr_inp = new_inputs[1]
            lr_inp = new_inputs[0]
            lr_sc_att_inp = new_inputs[0]  # separate var necessary for stack hr_fusion
        else:
            hr_inp = inputs[1]
            lr_inp = inputs[0]
            lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        
        for i in range(len(lr_sc_att_inp)):
            lr_sc_att_inp[i] = lr_sc_att_inp[i].detach()

        # debug
        # print("lr_input shape", lr_inp[0].shape)

        hr_scale = self.scales[1]
        lr_scale = self.scales[0]
        
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        if "CFFM" in self.head_type and "VideoAttn" in self.head_type:
            lr_seg, _c2 = self.head(lr_inp, return_feat = True)
            _c2 = _c2.detach()
        else:
            lr_seg = self.head(lr_inp)
            _c2 = None

        hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp, feat_video = _c2, lr_out = lr_seg.detach(), hr_out = hr_seg.detach())
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        # debug: save attn weight
        # for i_class in range(int(att.shape[1])):
        #     # this_map = (att[0, i_class:i_class+1, :, :].detach().cpu().numpy() * 255).astype(np.uint8)
        #     this_map = att[0, i_class:i_class+1, :, :].permute(1,2,0).detach().cpu().numpy()
        #     plt.imshow(this_map)
        #     plt.savefig(f"debug/attn_weights_{i_class}.png")

            # this_map = cv2.cvtColor(this_map,cv2.COLOR_GRAY2RGB)
            # cv2.imwrite(f"debug/attn_weights_{i_class}.png", this_map)
            # save_image(this_map, f"debug/attn_weights_{i_class}.png")

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        if self.debug_cnt % 100 == 0:
            seg_pred = torch.argmax(seg_logits[0], dim=1)
            seg_pred = 1.0*seg_pred / seg_pred.max()
            # save_image(seg_pred, 'debug/seg_pred_{}.png'.format(self.debug_cnt))
            # print("gt_semantic_seg shape", gt_semantic_seg.shape)
            # save_image(1.0*gt_semantic_seg / gt_semantic_seg.max(), 'debug/seg_gt_{}.png'.format(self.debug_cnt))
        self.debug_cnt += 1

        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""

        if seg_label.dim() == 5:
            seg_label = seg_label[:,-1]
        elif seg_label.shape[0] == self.num_clips:
            seg_label = seg_label[-1:]

        fused_seg, lr_seg, hr_seg = seg_logit

        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            # self.debug_output['Cropped GT'] = \
            #     cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        return loss
