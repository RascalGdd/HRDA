import os.path as osp
import torch
import mmcv
import numpy as np

from ..builder import PIPELINES

def generate_template_depth_mask(image_size, level_configs = np.arange(0,2,0.002)):
    H2, W2 = image_size[0]*2, image_size[1]*2
    H, W = image_size[0], image_size[1]
    num_levels = len(level_configs) + 1
    central_point = [H, W]
    depth_mask_template = np.zeros((H2, W2), dtype=float)
    print(depth_mask_template.shape)
    for level_scale in level_configs:
        x1_bias = int(H * level_scale / 2)
        x2_bias = int(W * level_scale / 2)
        x1_min = max(0, H - x1_bias)
        x1_max = min(H2, H + x1_bias)
        x2_min = max(0, W - x2_bias)
        x2_max = min(W2, W + x2_bias)
        depth_mask_template[x1_min:x1_max, x2_min:x2_max] += 1
    return depth_mask_template / num_levels

def vanishing_point_to_depth_mask(vanishing_mode, vanishing_point, image_size, level_configs = np.arange(0,2,0.002)):
# vanishing_points: tuple, in pixel
# image_size: tuple (H, W)
    if not hasattr(vanishing_point_to_depth_mask, "template"):
        vanishing_point_to_depth_mask.template = generate_template_depth_mask(image_size, level_configs) #(2H, 2W)

    H, W = image_size[0], image_size[1]
    if not vanishing_point:
        if vanishing_mode == "night":
            vanishing_point = [int(H/2), int(W/2)]
        elif vanishing_mode == "day":
            vanishing_point = [int((2*H) / 3), int(W / 2)]
    x1, x2 = int(vanishing_point[0]), int(vanishing_point[1])
    bias1, bias2 = int(x1-H/2), int(x2-W/2)
    c1, c2 = H+bias1, W+bias2
    x1_min = int(c1-H/2)
    x1_max = int(c1+H/2)
    x2_min = int(c2-W/2)
    x2_max = int(c2+W/2)
    return vanishing_point_to_depth_mask.template[x1_min:x1_max, x2_min:x2_max]

def get_global_pos_emb(image_size, emb_dim):
    if not hasattr(get_global_pos_emb, "image_size"):
        get_global_pos_emb.image_size = image_size
    if not hasattr(get_global_pos_emb, "pos") or image_size != get_global_pos_emb.image_size:
        temperature = 10000
        mask = torch.zeros((image_size[0],image_size[1]), dtype=int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        dim_t = torch.arange(emb_dim, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / emb_dim)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=3
        ).flatten(2)

        get_global_pos_emb.pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1) #(D, H, W)
        get_global_pos_emb.pos = get_global_pos_emb.pos.numpy()

    return get_global_pos_emb.pos

# def get_global_pos_emb(image_size):
#     if not hasattr(get_global_pos_emb, "image_size"):
#         get_global_pos_emb.image_size = image_size
#     if not hasattr(get_global_pos_emb, "pos_emb") or image_size != get_global_pos_emb.image_size:
#         total_size = int(image_size[0] * image_size[1])
#         idx_map = np.arange(total_size).astype(float)
#         get_global_pos_emb.pos_emb = np.zeros((image_size[0],image_size[1]), dtype=float)
#         for h in range(image_size[0]):
#             get_global_pos_emb.pos_emb[h,:] = idx_map[h*image_size[1]:(h+1)*image_size[1]]
#     return get_global_pos_emb.pos_emb

# def get_global_pos_emb(image_size):
#     if not hasattr(get_global_pos_emb, "image_size"):
#         get_global_pos_emb.image_size = image_size
#     if not hasattr(get_global_pos_emb, "pos_emb") or image_size != get_global_pos_emb.image_size:
#         total_size = int(image_size[0] * image_size[1])
#         idx_map = np.arange(total_size, dtype=float)
#         get_global_pos_emb.pos_emb = np.reshape(idx_map, (image_size[0],image_size[1]))
#     return get_global_pos_emb.pos_emb

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # add vanishing_mask here!

        if "night" in results['filename']:
            vanishing_mode = "night"
        else:
            vanishing_mode = "day"

        image_size = (img.shape[0], img.shape[1])
        vanishing_mask = vanishing_point_to_depth_mask(vanishing_mode, None, image_size)
        results["vanishing_mask"] = vanishing_mask.astype(np.float32)
        pos_emb = get_global_pos_emb(image_size, 64)
        results["pos_emb"] = pos_emb.astype(np.float32)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.
    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
