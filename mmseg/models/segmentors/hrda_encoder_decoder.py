# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image

from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    # print("offset_h", offset_h)
    # print("offset_w", offset_w)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

def get_crop_bbox_vanish_point(depth_map, crop_size, divisible=1):
    """Get a crop bounding box according to depth_map."""
    img_h, img_w = depth_map.shape[-2:]
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)

    # max_val: (B,1,1,1)
    # max_val = torch.amax(depth_map, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
    max_val = torch.max(depth_map).item()
    depth_map[depth_map == max_val] = 0
    max_val = torch.max(depth_map).item() # use the second-largest

    # max_val_ids: (L,4), e.g., [[0,0,0,0], [0,0,0,1],..., [3,0,16,501]] (list of max_value_points)
    max_val_ids = (depth_map == max_val).nonzero(as_tuple = False)

    central_id = max_val_ids.float().mean(dim=0).long()[2:]  # (h,w)
    if divisible:
        if central_id[0] % 4 != 0:
            central_id[0] = central_id[0] - central_id[0] % 4
        if central_id[1] % 4 != 0:
            central_id[1] = central_id[1] - central_id[1] % 4

    crop_y1 = (central_id[0] - crop_size[0] / 2).long().item()
    crop_y2 = (central_id[0] + crop_size[0] / 2).long().item()
    crop_x1 = (central_id[1] - crop_size[1] / 2).long().item()
    crop_x2 = (central_id[1] + crop_size[1] / 2).long().item()

    if crop_y1 < 0:
        crop_y2 += - crop_y1
        crop_y1 = 0
    if crop_y2 > img_h:
        crop_y1 -= crop_y2 - img_h
        crop_y2 = img_h
    if crop_x1 < 0:
        crop_x2 += - crop_x1
        crop_x1 = 0
    if crop_x2 > img_w:
        crop_x1 -= crop_x2 - img_w
        crop_x2 = img_w

    return crop_y1, crop_y2, crop_x1, crop_x2

def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img

# from https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :self.org_channels].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels

class DepthMapEmbedding(nn.Module):
    def __init__(self, emb_dim):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(DepthMapEmbedding, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, depth_map):
        return depth_map.repeat(1, self.emb_dim, 1, 1)


@SEGMENTORS.register_module()
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scales=[1],
                 hr_crop_size=None,
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 lr_only=False,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1,
                 pos_emb=False,
                 depthmap_emb=False
    ):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        if len(scales) == 1 and scales[0]<0:
            lr_only = True
        self.lr_only = lr_only
        decode_head.lr_only = lr_only

        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg
        )

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop

        self.debug_count = 0 # debug
        self.pos_emb_dim = self.decode_head.pos_emb_dim
        if self.pos_emb_dim > 0:
            self.pos_emb = PositionalEncodingPermute2D(self.pos_emb_dim)

        self.depthmap_emb_dim = self.decode_head.depthmap_emb_dim
        if self.depthmap_emb_dim > 0:
            self.depthmap_emb = DepthMapEmbedding(self.depthmap_emb_dim)
        # TODO: add depthmap emb

    def extract_unscaled_feat(self, img):
        x = self.backbone(img[:,:3,:,:]) # ensure input 3 channels
        if self.with_neck:
            x = self.neck(x)

        if self.pos_emb_dim > 0:
            x.append(self.pos_emb(img)) # pos emb
            print("Debug: pos embedding calculated in encoding!", x[-1].shape) # debug

        if self.depthmap_emb_dim > 0:
            x.append(self.depthmap_emb(img[:,3:,:,:])) # pos emb
            print("Debug: depthmap embedding calculated in encoding!", x[-1].shape) # debug

        return x

    def extract_slide_feat(self, img):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        mres_feats = []
        batch_size = img.shape[0]
        # assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        for i, s in enumerate(self.scales):
            scaled_img = resize(
                input=img,
                scale_factor=s,
                mode='bilinear',
                align_corners=self.align_corners
            )
            if self.crop_size is not None and i >= 1:
                crop_boxes = []
                scaled_imgs = []
                for b in range(batch_size):
                    depth_map = scaled_img[b:b+1,-1:,:,:]
                    crop_boxes.append(get_crop_bbox_vanish_point(depth_map, self.crop_size, self.crop_coord_divisible))
                    crop_box = crop_boxes[-1]
                    scaled_imgs.append(crop(scaled_img[b:b+1,0:3:,:,:], crop_box))
                    
                    # debug
                    # scaled_img_tmp = caled_imgs[-1]
                    # if self.debug_count < 100:
                    #     print("image shape:", img.shape)
                    #     print("cropped image shape:", scaled_img_tmp.shape)
                    #     save_image(img[b,:3,:,:], 'debug/{}_{}_ori_image.png'.format(self.debug_count, b))
                    #     save_image(img[b,3:,:,:], 'debug/{}_{}_depth_map.png'.format(self.debug_count, b))
                    #     print("crop box h1 h2 w1 w2:", crop_box)
                    #     save_image(scaled_img_tmp[b,:3,:,:], 'debug/{}_{}_cropped_image.png'.format(self.debug_count, b))
                    #     crop_mask = img[b,3:,:,:] * 0
                    #     crop_y1, crop_y2, crop_x1, crop_x2 = crop_box
                    #     crop_mask[:, crop_y1:crop_y2, crop_x1:crop_x2] = 1
                    #     save_image(crop_mask, 'debug/{}_{}_crop_mask.png'.format(self.debug_count, b))
                    #     for feat in mres_feats[-1]:
                    #         print("mres feat shapes:", feat.shape)
                    # else:
                    #     break_debug

                self.debug_count += 1
                scaled_img = torch.cat(scaled_imgs, dim=0)
                self.decode_head.set_batch_hr_crop_box(crop_boxes)
                mres_feats.append(self.extract_unscaled_feat(scaled_img))

            elif i >= 1 and self.hr_slide_inference:
                mres_feats.append(self.extract_slide_feat(scaled_img))

            else:
                mres_feats.append(self.extract_unscaled_feat(scaled_img))

        out = self._decode_head_forward_test(mres_feats, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _forward_train_features(self, img):
        mres_feats = []
        self.decode_head.debug_output = {}
        # assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        return mres_feats, prob_vis

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        loss_decode = self._decode_head_forward_train(mres_feats, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        return losses

    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}
