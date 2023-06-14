import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES

@PIPELINES.register_module()
class AlignedResize_clips(object):
    """Resize images & seg.
    
    This class is based on ``Resize``, the only difference is 
    it ensure the long and short sides are divisible by ``size_divisor``.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 size_divisor=32):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.size_divisor = size_divisor

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """
        assert isinstance(results['img'], list)

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'][0].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _align(self, img, size_divisor, interpolation=None):
        align_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
        align_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor
        if interpolation == None:
            img = mmcv.imresize(img, (align_w, align_h))
        else:
            img = mmcv.imresize(img, (align_w, align_h), interpolation=interpolation)
        return img

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        assert isinstance(results['img'], list)
        img_all=[]
        if self.keep_ratio:
            for im_one in results['img']:
                img, scale_factor = mmcv.imrescale(
                    im_one, results['scale'], return_scale=True)
                #### align ####
                img = self._align(img, self.size_divisor)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                img_all.append(img)
            new_h, new_w = img.shape[:2]
            h, w = results['img'][0].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            for im_one in results['img']:
                img, w_scale, h_scale = mmcv.imresize(
                    im_one, results['scale'], return_scale=True)
                img_all.append(img)

            h, w = img.shape[:2]
            assert int(np.ceil(h / self.size_divisor)) * self.size_divisor == h and \
                   int(np.ceil(w / self.size_divisor)) * self.size_divisor == w, \
                   "img size not align. h:{} w:{}".format(h,w)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img_all
        results['img_shape'] = img_all[0].shape
        results['pad_shape'] = img_all[0].shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            assert isinstance(results[key], list)
            gt_seg_all=[]
            if self.keep_ratio:
                for gt_one in results[key]:
                    gt_seg = mmcv.imrescale(
                        gt_one, results['scale'], interpolation='nearest')
                    gt_seg = self._align(gt_seg, self.size_divisor, interpolation='nearest')
                    gt_seg_all.append(gt_seg)
            else:
                for gt_one in results[key]:
                    gt_seg = mmcv.imresize(
                        gt_one, results['scale'], interpolation='nearest')
                    gt_seg_all.append(gt_seg)
                h, w = gt_seg.shape[:2]
                assert int(np.ceil(h / self.size_divisor)) * self.size_divisor == h and \
                       int(np.ceil(w / self.size_divisor)) * self.size_divisor == w, \
                    "gt_seg size not align. h:{} w:{}".format(h, w)
            results[key] = gt_seg_all

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class RandomFlip_clips(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            img_all=[]
            assert isinstance(results['img'], list)
            for im_one in results['img']:
                img = mmcv.imflip(
                    im_one, direction=results['flip_direction'])
                img_all.append(img)
            results['img']=img_all

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                assert isinstance(results[key], list)
                gt_seg_all=[]
                for gt in results[key]:
                    gt_flip = mmcv.imflip(
                        gt, direction=results['flip_direction']).copy()
                    gt_seg_all.append(gt_flip)
                results[key]=gt_seg_all
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

@PIPELINES.register_module()
class Pad_clips(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img_clips(self, results):
        """Pad images according to ``self.size``."""
        assert isinstance(results['img'], list)
        img_all=[]
        if self.size is not None:
            for im in results['img']:
                padded_img = mmcv.impad(
                    im, shape=self.size, pad_val=self.pad_val)
                img_all.append(padded_img)
        elif self.size_divisor is not None:
            for im in results['img']:
                padded_img = mmcv.impad_to_multiple(
                    im, self.size_divisor, pad_val=self.pad_val)
                img_all.append(padded_img)
        results['img'] = img_all
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg_clips(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            assert isinstance(results[key], list)
            gt_seg_all=[]
            for gt in results[key]:
                gt_seg= mmcv.impad(
                    gt,
                    shape=results['pad_shape'][:2],
                    pad_val=self.seg_pad_val)
                gt_seg_all.append(gt_seg)

            results[key]=gt_seg_all

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img_clips(results)
        self._pad_seg_clips(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class Normalize_clips(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        assert isinstance(results['img'], list)
        img_all=[]
        for im in results['img']:
            img = mmcv.imnormalize(im, self.mean, self.std,
                                          self.to_rgb)
            img_all.append(img)

        results['img']=img_all
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class RandomCrop_clips(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert isinstance(results['img'], list)
        img = results['img'][-1]
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'][-1], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img_all=[]
        for im_one in results['img']:
            img = self.crop(im_one, crop_bbox)
            img_all.append(img)
        img_shape = img.shape
        results['img'] = img_all
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            assert isinstance(results[key], list)
            gt_seg_all=[]
            for gt_one in results[key]:
                gt_seg = self.crop(gt_one, crop_bbox)
                gt_seg_all.append(gt_seg)
            results[key]=gt_seg_all

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class PhotoMetricDistortion_clips(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert isinstance(results['img'],list)
        img_all=[]
        for im in results['img']:
            # img = results['img']
            # random brightness
            img = self.brightness(im)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                img = self.contrast(img)

            # random saturation
            img = self.saturation(img)

            # random hue
            img = self.hue(img)

            # random contrast
            if mode == 0:
                img = self.contrast(img)

            img_all.append(img)

        results['img'] = img_all
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


