import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size, window_size, C) # debug
    return windows

def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows

def window_partition_noreshape_overlap(x, window_size, stride):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x2 = x.permute(0, 3, 1, 2)
    n_windows_h = int((H - window_size) / stride) + 1
    n_windows_w = int((W - window_size) / stride) + 1
    windows = F.unfold(x2, (window_size, window_size),
        dilation=1, padding=0, stride=stride
    ).permute(0,2,1) # (B, n_windows, C*window_size*window_size)
    windows = windows.view(B, n_windows_h, n_windows_w, window_size, window_size, C)
    return windows

@torch.no_grad()
def window_partition_vanishing_point_no_reshape(x, vp_roi, window_size, stride = None):
    B, H, W, C = x.shape
    [hmin, hmax, wmin, wmax] = vp_roi
    if stride is None:
        stride = window_size
    else:
        assert type(stride) == int
    return window_partition_noreshape_overlap(x[:,hmin:hmax,wmin:wmax,:], window_size, stride)

def window_partition_vanishing_point(x, vp_roi, window_size, stride = None, sel_n_windows = None):
    B, H, W, C = x.shape
    roi_windows = window_partition_vanishing_point_no_reshape(
        x, vp_roi, window_size, stride
    ) # B, n_windows_h, n_windows_w, window_size, window_size, C
    roi_windows = roi_windows.view(B, -1, window_size, window_size, C)
    if sel_n_windows is not None:
        assert type(sel_n_windows) == int
        n_windows_h = int(math.sqrt(roi_windows.shape[1]))
        roi_windows = roi_windows.view(B, n_windows_h, n_windows_h, window_size, window_size, -1)
        assert sel_n_windows <= n_windows_h
        margin = int((n_windows_h - sel_n_windows) / 2)
        if margin > 0:
            roi_windows = roi_windows[B, :, margin:margin + sel_n_windows, margin + sel_n_windows]
        roi_windows = roi_windows.reshape(-1, window_size, window_size, C)
    return roi_windows.view(-1, window_size, window_size, C)

def window_partition_all(x, vp_roi, window_size, stride = None, sel_n_windows = None):
    B, H, W, C = x.shape
    default_windows = window_partition(x, window_size)
    roi_windows = window_partition_vanishing_point(x, vp_roi, window_size, stride)
    if sel_n_windows is not None:
        assert type(sel_n_windows) == int
        n_windows_h = int(math.sqrt(roi_windows.shape[0]))
        roi_windows = roi_windows.view(n_windows_h, n_windows_h,  window_size, window_size, -1)
        assert sel_n_windows <= n_windows_h
        margin = int((n_windows_h - sel_n_windows)/2)
        if margin > 0:
            roi_windows = roi_windows[margin:margin+sel_n_windows, margin+sel_n_windows]
        roi_windows = roi_windows.reshape(-1, window_size, window_size, C)
    return torch.cat([default_windows, roi_windows], dim = 1)

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_roll_masks(H, W, window_size, shift_size):
    #####################################
    # move to top-left
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, H-window_size),
                slice(H-window_size, H-shift_size),
                slice(H-shift_size, H))
    w_slices = (slice(0, W-window_size),
                slice(W-window_size, W-shift_size),
                slice(W-shift_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_tl = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    ####################################
    # move to top right
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, H-window_size),
                slice(H-window_size, H-shift_size),
                slice(H-shift_size, H))
    w_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_tr = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    ####################################
    # move to bottom left
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, H))
    w_slices = (slice(0, W-window_size),
                slice(W-window_size, W-shift_size),
                slice(W-shift_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_bl = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    ####################################
    # move to bottom right
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, H))
    w_slices = (slice(0, shift_size),
                slice(shift_size, window_size),
                slice(window_size, W))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask_br = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    # append all
    attn_mask_all = torch.cat((attn_mask_tl, attn_mask_tr, attn_mask_bl, attn_mask_br), -1)
    return attn_mask_all

def generate_template_depth_mask(image_size, level_configs = np.arange(0,2,0.002)):
    H2, W2 = image_size[0]*2, image_size[1]*2
    H, W = image_size[0], image_size[1]
    num_levels = len(level_configs) + 1
    central_point = [H, W]
    depth_mask_template = torch.zeros((H2, W2)).to(float)
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

def vanishing_point_to_depth_mask(image_size, level_configs = np.arange(0,2,0.002)):
    template = generate_template_depth_mask(image_size, level_configs) #(2H, 2W)

    H, W = image_size[0], image_size[1]
    vanishing_point = [int(2*H/3), int(3*W/4)]
    x1, x2 = int(vanishing_point[0]), int(vanishing_point[1])
    bias1, bias2 = int(x1-H/2), int(x2-W/2)
    c1, c2 = H+bias1, W+bias2
    x1_min = int(c1-H/2)
    x1_max = int(c1+H/2)
    x2_min = int(c2-W/2)
    x2_max = int(c2+W/2)
    return template[x1_min:x1_max, x2_min:x2_max]

def get_vanishing_point(
        vanishing_map, n_windows_h = 3, n_windows_w = 3, window_size = 7,
        align_to_center = True, stride = None, extra_n_windows = 0
):
    B, _, H, W = vanishing_map.shape # Debug: vanishing_map shape may vary in real use, this is only debug mode

    # max_val: (B,1,1,1)
    max_val = torch.amax(vanishing_map, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)

    # max_val_ids: (L,4), e.g., [[0,0,0,0], [0,0,0,1],..., [3,0,16,501]] (list of max_value_points)
    max_val_ids = (vanishing_map == max_val).nonzero()
    central_id = max_val_ids.float().mean(dim=0).long()[2:]  # (h,w)

    if align_to_center:
        half_window_size = int(window_size / 2)
        if n_windows_h % 2 == 0:
            central_id[0] = torch.round(central_id[0] / window_size) * window_size
        else:
            central_id[0] = torch.round((central_id[0] - half_window_size) / window_size) * window_size + half_window_size
        if n_windows_w % 2 == 0:
            central_id[1] = torch.round(central_id[1] / window_size) * window_size
        else:
            central_id[1] = torch.round((central_id[1] - half_window_size) / window_size) * window_size + half_window_size

    if stride is not None:
        assert type(stride) == int
        h_total_len = int((n_windows_h-1)*stride + window_size)
        w_total_len = int((n_windows_w-1)*stride + window_size)
    else:
        h_total_len = int(n_windows_h*window_size)
        w_total_len = int(n_windows_w*window_size)
        stride = window_size

    h_len_left = int(h_total_len / 2) + extra_n_windows * stride
    h_len_right = h_total_len - int(h_total_len / 2) + extra_n_windows * stride
    w_len_top = int(w_total_len / 2) + extra_n_windows * stride
    w_len_dwn = w_total_len - int(w_total_len / 2) + extra_n_windows * stride
    hmin = central_id[0] - h_len_left
    hmax = central_id[0] + h_len_right
    wmin = central_id[1] - w_len_top
    wmax = central_id[1] + w_len_dwn
    if hmin < 0:
        hmax += - hmin
        central_id[0] += - hmin
        hmin = 0
    if hmax > H:
        hmin -= hmax - H
        central_id[0] -= hmax - H
        hmax = H
    if wmin < 0:
        wmax += - wmin
        central_id[1] += - wmin
        wmin = 0
    if wmax > W:
        wmin -= wmax - W
        central_id[1] -= wmax - W
        wmax = W

    return central_id, [hmin,hmax,wmin,wmax]

def get_central_window_id_roi(central_id, window_size, n_windows):
# n_windows is the number of windows per edge
    half_window_size = int(window_size / 2)
    assert central_id[0] % window_size == 0 or central_id[0] % window_size == half_window_size
    assert central_id[1] % window_size == 0 or central_id[1] % window_size == half_window_size
    if central_id[0] % window_size == 0 and central_id[1] % window_size == 0:
        assert n_windows % 2 == 0
        central_window_id = (central_id[0] / window_size, central_id[1] / window_size)
        half_n_windows = n_windows / 2
        hmin_window_id = int(central_window_id[0] - half_n_windows)
        hmax_window_id = int(central_window_id[0] + half_n_windows)
        wmin_window_id = int(central_window_id[1] - half_n_windows)
        wmax_window_id = int(central_window_id[1] + half_n_windows)

    elif central_id[0] % window_size == half_window_size and central_id[1] % window_size == half_window_size:
        assert n_windows % 2 == 1
        central_window_id = (
            (central_id[0]+half_window_size) / window_size,
            (central_id[1]+half_window_size) / window_size
        )
        half_n_windows = (n_windows - 1) / 2
        hmin_window_id = int(central_window_id[0] - half_n_windows - 1)
        hmax_window_id = int(central_window_id[0] + half_n_windows)
        wmin_window_id = int(central_window_id[1] - half_n_windows - 1)
        wmax_window_id = int(central_window_id[1] + half_n_windows)
    else:
        assert 0
    return [hmin_window_id, hmax_window_id, wmin_window_id, wmax_window_id]

def get_relative_position_index(q_windows, k_windows):
    """
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)

    Returns:
        relative_position_index: query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += k_windows[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[1] + k_windows[1]) - 1
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index

def get_relative_position_index3d(q_windows, k_windows, num_clips):
    """
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)

    Returns:
        relative_position_index: query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_d_q = torch.arange(num_clips)
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_d_q, coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_d_k = torch.arange(num_clips)
    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_d_k, coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += num_clips - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[0] - 1
    relative_coords[:, :, 2] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[0] + k_windows[0] - 1)*(q_windows[1] + k_windows[1] - 1)
    relative_coords[:, :, 1] *= (q_windows[1] + k_windows[1] - 1)
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index


class WindowAttention3d3(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        expand_size (int): The expand size at focal level 1.
        window_size (tuple[int]): The height and width of the window.
        focal_window (int): Focal region size.
        focal_level (int): Focal attention level.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0 
        pool_method (str): window pooling method. Default: none
    """

    def __init__(self, dim, expand_size, window_size, focal_window, focal_level, num_heads, 
                    qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", focal_l_clips=[7,1,2], focal_kernel_clips=[7,5,3]):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        # define a parameter table of relative position bias for each window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.vp_n_windows = 8
        self.vp_roi_n_windows = self.vp_n_windows * self.vp_n_windows
        self.vp_stride = 3
        self.vp_unfold_stride_clips = [1, 2, 4]
        self.vp_n_windows_ori = 4
        self.vp_roi_n_windows_ori = self.vp_n_windows_ori * self.vp_n_windows_ori

        if self.expand_size > 0 and focal_level > 0:
            # define a parameter table of position bias between window and its fine-grained surroundings
            self.window_size_of_key = self.window_size[0] * self.window_size[1] if self.expand_size == 0 else \
                (4 * self.window_size[0] * self.window_size[1] - 4 * (self.window_size[0] -  self.expand_size) * (self.window_size[0] -  self.expand_size))        
            self.relative_position_bias_table_to_neighbors = nn.Parameter(
                torch.zeros(1, num_heads, self.window_size[0] * self.window_size[1], self.window_size_of_key))  # Wh*Ww, nH, nSurrounding
            trunc_normal_(self.relative_position_bias_table_to_neighbors, std=.02)

            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size, :-self.expand_size] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size, self.expand_size:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size:, :-self.expand_size] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size:, self.expand_size:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", mask_rolled.nonzero().view(-1))

        if pool_method != "none" and focal_level > 1:
            self.relative_position_bias_table_to_windows = nn.ParameterList()
            self.relative_position_bias_table_to_windows_clips = nn.ParameterList()
            self.relative_position_bias_table_to_windows_clips_vp = nn.ParameterList()
            self.unfolds = nn.ModuleList()
            self.unfolds_clips=nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level-1):
                stride = 2**k    
                kernel_size = 2*(self.focal_window // 2) + 2**k + (2**k-1)
                # define unfolding operations                
                self.unfolds += [nn.Unfold(
                    kernel_size=(kernel_size, kernel_size), 
                    stride=stride, padding=kernel_size // 2)
                ]

                # define relative position bias table
                relative_position_bias_table_to_windows = nn.Parameter(
                    torch.zeros(
                        self.num_heads,
                        (self.window_size[0] + self.focal_window + 2**k - 2) * (self.window_size[1] + self.focal_window + 2**k - 2), 
                        )
                )
                trunc_normal_(relative_position_bias_table_to_windows, std=.02)
                self.relative_position_bias_table_to_windows.append(relative_position_bias_table_to_windows)

                # define relative position bias index
                relative_position_index_k = get_relative_position_index(self.window_size, to_2tuple(self.focal_window + 2**k - 1))
                # relative_position_index_k = get_relative_position_index3d(self.window_size, to_2tuple(self.focal_window + 2**k - 1), num_clips)
                self.register_buffer("relative_position_index_{}".format(k), relative_position_index_k)

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size, kernel_size); mask[(2**k)-1:, (2**k)-1:] = 1
                    self.register_buffer("valid_ind_unfold_{}".format(k), mask.flatten(0).nonzero().view(-1))

            for k in range(len(focal_l_clips)): # [1,2,3]
                # kernel_size=focal_kernel_clips[k]
                focal_l_big_flag=False
                if focal_l_clips[k]>self.window_size[0]:
                    stride=1
                    padding=0
                    kernel_size=focal_kernel_clips[k] # [7,5,3]
                    kernel_size_true=kernel_size
                    focal_l_big_flag=True
                    # stride=math.ceil(self.window_size/focal_l_clips[k])
                    # padding=(kernel_size-stride)/2
                else:
                    stride = focal_l_clips[k] 
                    # kernel_size   
                    # kernel_size = 2*(focal_kernel_clips[k]// 2) + 2**focal_l_clips[k] + (2**focal_l_clips[k]-1)
                    kernel_size = focal_kernel_clips[k]     ## kernel_size must be jishu 
                    assert kernel_size%2==1
                    padding=kernel_size // 2
                    # kernel_size_true=focal_kernel_clips[k]+2**focal_l_clips[k]-1
                    kernel_size_true=kernel_size
                # stride=math.ceil(self.window_size/focal_l_clips[k])
                
                self.unfolds_clips += [nn.Unfold(
                    kernel_size=(kernel_size, kernel_size), 
                    stride=stride,
                    padding=padding)
                ]
                relative_position_bias_table_to_windows = nn.Parameter(
                    torch.zeros(
                        self.num_heads,
                        (self.window_size[0] + kernel_size_true - 1) * (self.window_size[0] + kernel_size_true - 1),
                        )
                )
                n_poses = 9
                self.n_poses_mapping = {-4: 0,
                                   -3: 1,
                                   -2: 2,
                                   -1: 3,
                                    0: 4,
                                    1: 5,
                                    2: 6,
                                    3: 7,
                                    4: 8}
                relative_position_bias_table_to_windows_vp = nn.Parameter(
                    torch.zeros(n_poses,
                        self.num_heads,
                        self.window_size[0]**2, kernel_size_true**2,
                        )
                )
                trunc_normal_(relative_position_bias_table_to_windows, std=.02)
                trunc_normal_(relative_position_bias_table_to_windows_vp, std=.02)
                self.relative_position_bias_table_to_windows_clips.append(relative_position_bias_table_to_windows)
                self.relative_position_bias_table_to_windows_clips_vp.append(relative_position_bias_table_to_windows_vp)
                relative_position_index_k = get_relative_position_index(self.window_size, to_2tuple(kernel_size_true))
                self.register_buffer("relative_position_index_clips_{}".format(k), relative_position_index_k)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # added for vanishing point
        self.vanishing_point_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        self.attn_drop_vp = nn.Dropout(attn_drop)
        self.norm_vp = nn.LayerNorm(dim)
        self.proj_vp = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.focal_l_clips=focal_l_clips
        self.focal_kernel_clips=focal_kernel_clips

    def forward(self, x_all, mask_all=None, batch_size=None, num_clips=None):
        """
        Args:
            x_all (list[Tensors]): input features at different granularity
            mask_all (list[Tensors/None]): masks for input features at different granularity
        """
        x = x_all[0][0] # 

        B0, nH, nW, C = x.shape
        # assert B==batch_size*num_clips
        vanishing_map = vanishing_point_to_depth_mask((nH, nW)).unsqueeze(0).unsqueeze(0)
        #debug: this is only for debug, in real use, depth map is argument
        assert B0==batch_size
        qkv = self.qkv(x).reshape(B0, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B0, nH, nW, C

        central_id, [hmin, hmax, wmin, wmax] = get_vanishing_point(
            vanishing_map, self.vp_n_windows, self.vp_n_windows, self.window_size[0], stride=self.vp_stride
        )
        vp_roi = [hmin, hmax, wmin, wmax]

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(
                t, self.window_size[0]
            ).view(
                -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2), 
            (q, k, v)
        ) # ((190, 8, 49, 32), (190, 8, 49, 32), (190, 8, 49, 32))

        k_windows_vp, v_windows_vp = map(
            lambda t: window_partition_vanishing_point(
                t, vp_roi, self.window_size[0], stride = self.vp_stride, sel_n_windows = None
            ).view(
                B0, -1, self.window_size[0] * self.window_size[0], self.dim
            ),
            (k, v)
        ) # ((B, 64, 49, 256), (B, 64, 49, 256))

        if self.expand_size > 0 and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_br, v_br) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
            )        
            
            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads), 
                (k_tl, k_tr, k_bl, k_br)
            )
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads), 
                (v_tl, v_tr, v_bl, v_br)
            ) # 70=H, 133=W
            k_rolled = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 1).transpose(1, 2)
            v_rolled = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 1).transpose(1, 2) # (190,8,196,32)
            
            # mask out tokens in current window
            # print("self.valid_ind_rolled.shape: ", self.valid_ind_rolled.shape)    # [132]
            # print("k_rolled.shape: ", k_rolled.shape)    # [1296, 8, 196, 32]
            k_rolled = k_rolled[:, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, self.valid_ind_rolled] # (190,8,132,32)
            # (9,8,132,32)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2) # (190,8,181,32)
        else:
            k_rolled = k_windows; v_rolled = v_windows; 

        # print("k_rolled.shape: ", k_rolled.shape)  # [1296, 8, 181, 32]

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level-1):
                stride = 2**k
                x_window_pooled = x_all[0][k+1]  # B0, nWh, nWw, C
                nWh, nWw = x_window_pooled.shape[1:3] # 10, 19

                # generate mask for pooled windows
                # print("x_window_pooled.shape: ", x_window_pooled.shape)
                mask = x_window_pooled.new(nWh, nWw).fill_(1) # (10,19)
                # print("here: ",x_window_pooled.shape, self.unfolds[k].kernel_size, self.unfolds[k](mask.unsqueeze(0).unsqueeze(1)).shape)
                # print(mask.unique())
                unfolded_mask = self.unfolds[k](mask.unsqueeze(0).unsqueeze(1)).view(
                    1, 1, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(0, 4, 2, 3, 1).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1) # (190, 25, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                # print("unfolded_mask.shape: ", unfolded_mask.shape, unfolded_mask.unique())
                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0) # (1,190,25)
                # print((x_window_masks == 0).sum(), (x_window_masks > 0).sum(), x_window_masks.unique())
                x_window_masks = x_window_masks.masked_fill(x_window_masks == 0, float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))  
                # print(x_window_masks.shape)          
                mask_all[0][k+1] = x_window_masks

                # generate k and v for pooled windows                
                qkv_pooled = self.qkv(x_window_pooled).reshape(B0, nWh, nWw, 3, C).permute(3, 0, 4, 1, 2).contiguous()
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]  # B0, C, nWh, nWw
                # (1,256,10,19), (1,256,10,19)

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k](t).view(
                    B0, C, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(0, 4, 2, 3, 1).contiguous().\
                    view(-1, self.unfolds[k].kernel_size[0]*self.unfolds[k].kernel_size[1], self.num_heads, C // self.num_heads).transpose(1, 2), 
                    (k_pooled_k, v_pooled_k)  # (B0 x (nH*nW)) x nHeads x (unfold_wsize x unfold_wsize) x head_dim
                ) # (190,8,25,32), (190,8,25,32)

                # print("k_pooled_k.shape: ", k_pooled_k.shape)
                # print("valid_ind_unfold_k.shape: ", valid_ind_unfold_k.shape)

                if k > 0:                    
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, valid_ind_unfold_k], (k_pooled_k, v_pooled_k)
                    )

                # print("k_pooled_k.shape: ", k_pooled_k.shape)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k] # [(190,8,25,32)]

            
            coords_q_moving_all = []
            coords_q_moving_opposite_all = []
            mask_coord_list_all = []
            mask_coord_list_opposite_all = []
            for k in range(len(self.focal_l_clips)):
                focal_l_big_flag=False
                if self.focal_l_clips[k]>self.window_size[0]:
                    stride=1
                    focal_l_big_flag=True
                else:
                    stride = self.focal_l_clips[k]
                # if self.window_size>=focal_l_clips[k]:
                #     stride=math.ceil(self.window_size/focal_l_clips[k])
                #     # padding=(kernel_size-stride)/2
                # else:
                #     stride=1
                    # padding=0
                x_window_pooled = x_all[k+1]
                nWh, nWw = x_window_pooled.shape[1:3] 
                mask = x_window_pooled.new(nWh, nWw).fill_(1)

                # import pdb; pdb.set_trace()
                # print(x_window_pooled.shape, self.unfolds_clips[k].kernel_size, self.unfolds_clips[k](mask.unsqueeze(0).unsqueeze(1)).shape)

                unfolded_mask = self.unfolds_clips[k](mask.unsqueeze(0).unsqueeze(1)).view(
                    1, 1, self.unfolds_clips[k].kernel_size[0], self.unfolds_clips[k].kernel_size[1], -1).permute(0, 4, 2, 3, 1).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1) # (190,49,1)

                # if (not focal_l_big_flag) and self.focal_l_clips[k]>0:
                #     valid_ind_unfold_k = getattr(self, "valid_ind_unfold_clips_{}".format(k))
                #     unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                # print("unfolded_mask.shape: ", unfolded_mask.shape, unfolded_mask.unique())
                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                # print((x_window_masks == 0).sum(), (x_window_masks > 0).sum(), x_window_masks.unique())
                x_window_masks = x_window_masks.masked_fill(x_window_masks == 0, float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))  
                # print(x_window_masks.shape)          
                mask_all[k+1] = x_window_masks

                # generate k and v for pooled windows                
                qkv_pooled = self.qkv(x_window_pooled).reshape(B0, nWh, nWw, 3, C).permute(3, 0, 4, 1, 2).contiguous()
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]  # B0, C, nWh, nWw

                #    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      motion VP moving direction          <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds_clips[k](t).view(B0, C, self.unfolds_clips[k].kernel_size[0], self.unfolds_clips[k].kernel_size[1], x_all[0][1].shape[1], x_all[0][1].shape[2]),
                    (k_pooled_k, v_pooled_k)  # (B0 x (nH*nW)) x nHeads x (unfold_wsize x unfold_wsize) x head_dim
                )

                matrix = torch.zeros([k_pooled_k.shape[-2], k_pooled_k.shape[-1]]).type(torch.float32)
                mask_coord_matrix = torch.arange((k_pooled_k.shape[-2]*k_pooled_k.shape[-1])).view((k_pooled_k.shape[-2], k_pooled_k.shape[-1]))
                k_matrix_moved = torch.zeros_like(k_pooled_k)
                k_matrix_moved_opposite = torch.zeros_like(k_pooled_k)
                v_matrix_moved = torch.zeros_like(v_pooled_k)
                v_matrix_moved_opposite = torch.zeros_like(v_pooled_k)
                coords_h_q = torch.arange(matrix.shape[0])
                coords_w_q = torch.arange(matrix.shape[1])
                coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))

                # assume vanishing point at the middle of the image TODO
                vp_coordinate = [matrix.shape[0] // 2, matrix.shape[1] // 2]
                relative_matrix = torch.zeros_like(coords_q)
                relative_matrix_tan = torch.zeros_like(matrix)
                relative_matrix_move = torch.zeros_like(coords_q)
                relative_matrix_move_opposite = torch.zeros_like(coords_q)
                delta_y_matrix = vp_coordinate[0] - coords_q[0]
                delta_x_matrix = vp_coordinate[1] - coords_q[1]
                relative_matrix_tan = delta_y_matrix/(delta_x_matrix + 0.0001)

                move_dir_90deg = ((relative_matrix_tan <= -2.414) + (relative_matrix_tan >= 2.414)).to(bool)
                relative_matrix_move[:, move_dir_90deg] += torch.tensor([1, 0]).unsqueeze(-1)
                move_dir_0deg = ((relative_matrix_tan <= 0.414) * (relative_matrix_tan >= -0.414)).to(bool)
                relative_matrix_move[:, move_dir_0deg] += torch.tensor([0, 1]).unsqueeze(-1)
                move_dir_45deg = ((relative_matrix_tan <= 2.414) * (relative_matrix_tan >= 0.414)).to(bool)
                relative_matrix_move[:, move_dir_45deg] += torch.tensor([1, 1]).unsqueeze(-1)
                move_dir_135deg = ((relative_matrix_tan <= -0.414) * (relative_matrix_tan >= -2.414)).to(bool)
                relative_matrix_move[:, move_dir_135deg] += torch.tensor([1, -1]).unsqueeze(-1)

                relative_matrix_move_opposite = - relative_matrix_move
                relative_matrix_move[0, 0][relative_matrix_move[0, 0, :] <= 0] = 0
                relative_matrix_move[0, -1][relative_matrix_move[0, -1, :] >= 0] = 0
                relative_matrix_move[1, :, 0][relative_matrix_move[1, :, 0] <= 0] = 0
                relative_matrix_move[1, :, -1][relative_matrix_move[1, :, -1] >= 0] = 0
                # print(relative_matrix_tan)
                relative_matrix_move_opposite[0, 0][relative_matrix_move_opposite[0, 0, :] <= 0] = 0
                relative_matrix_move_opposite[0, -1][relative_matrix_move_opposite[0, -1, :] >= 0] = 0
                relative_matrix_move_opposite[1, :, 0][relative_matrix_move_opposite[1, :, 0] <= 0] = 0
                relative_matrix_move_opposite[1, :, -1][relative_matrix_move_opposite[1, :, -1] >= 0] = 0

                coords_q_moved = coords_q + relative_matrix_move
                coords_q_moved_opposite = coords_q + relative_matrix_move_opposite

                mask_coord_list = (coords_q_moved[0] * coords_q_moved.shape[2] + coords_q_moved[1]).flatten().type(torch.long)
                mask_coord_list_opposite = (coords_q_moved_opposite[0] * coords_q_moved_opposite.shape[2] + coords_q_moved_opposite[1]).flatten().type(
                    torch.long)


                (k_pooled_k, v_pooled_k) = map(
                    lambda t: t.view(
                    B0, C, self.unfolds_clips[k].kernel_size[0], self.unfolds_clips[k].kernel_size[1], -1).permute(0, 4, 2, 3, 1).contiguous().\
                    view(-1, self.unfolds_clips[k].kernel_size[0]*self.unfolds_clips[k].kernel_size[1], self.num_heads, C // self.num_heads).transpose(1, 2),
                    (k_pooled_k, v_pooled_k)  # (B0 x (nH*nW)) x nHeads x (unfold_wsize x unfold_wsize) x head_dim
                )
                k_matrix_moved = k_pooled_k[mask_coord_list]
                k_matrix_moved_opposite = k_pooled_k[mask_coord_list_opposite]
                v_matrix_moved = v_pooled_k[mask_coord_list]
                v_matrix_moved_opposite = v_pooled_k[mask_coord_list_opposite]


                coords_q_moving_all += [relative_matrix_move]
                coords_q_moving_opposite_all += [relative_matrix_move_opposite]
                mask_coord_list_all += [mask_coord_list]
                mask_coord_list_opposite_all += [mask_coord_list_opposite]

                k_pooled_k = torch.cat([k_pooled_k, k_matrix_moved, k_matrix_moved_opposite], dim=2)
                v_pooled_k = torch.cat([v_pooled_k, v_matrix_moved, v_matrix_moved_opposite], dim=2)
                #    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      motion VP moving direction finish        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k] # [(190,8,49,32), (190,8,25,32), (190,8,9,32)]


            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2) # (190,8,181+25+49+25+9=289,32)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale

        attn = (q_windows @ k_all.transpose(-2, -1))  # B0*nW, nHead, window_size*window_size, focal_window_size*focal_window_size
        # (190,8,49,289)

        window_area = self.window_size[0] * self.window_size[1]    # 49
        # window_area_clips= num_clips*self.window_size[0] * self.window_size[1]       
        window_area_rolled = k_rolled.shape[2]

        # add relative position bias for tokens inside window
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (8,49,49); nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias.shape, attn.shape)
        attn[:, :, :window_area, :window_area] = attn[:, :, :window_area, :window_area] + relative_position_bias.unsqueeze(0)

        # add relative position bias for patches inside a window
        if self.expand_size > 0 and self.focal_level > 0:
            attn[:, :, :window_area, window_area:window_area_rolled] = attn[:, :, :window_area, window_area:window_area_rolled] + self.relative_position_bias_table_to_neighbors

        if self.pool_method != "none" and self.focal_level > 1:
            # add relative position bias for different windows in an image        
            offset = window_area_rolled
            # print(offset)
            for k in range(self.focal_level-1):
                # add relative position bias
                relative_position_index_k = getattr(self, 'relative_position_index_{}'.format(k))
                relative_position_bias_to_windows = self.relative_position_bias_table_to_windows[k][:, relative_position_index_k.view(-1)].view(
                    -1, self.window_size[0] * self.window_size[1], (self.focal_window+2**k-1)**2,
                ) # nH, NWh*NWw,focal_region*focal_region
                attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] = \
                    attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] + relative_position_bias_to_windows.unsqueeze(0)
                # add attentional mask
                if mask_all[0][k+1] is not None:
                    attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] = \
                        attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] + \
                            mask_all[0][k+1][:, :, None, None, :].repeat(attn.shape[0] // mask_all[0][k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[0][k+1].shape[-1])
                    
                offset += (self.focal_window+2**k-1)**2  
            # print(offset)
            for k in range(len(self.focal_l_clips)):
                focal_l_big_flag=False
                if self.focal_l_clips[k]>self.window_size[0]:
                    stride=1
                    padding=0
                    kernel_size=self.focal_kernel_clips[k]
                    kernel_size_true=kernel_size
                    focal_l_big_flag=True
                    # stride=math.ceil(self.window_size/focal_l_clips[k])
                    # padding=(kernel_size-stride)/2
                else:
                    stride = self.focal_l_clips[k] 
                    # kernel_size
                    # kernel_size = 2*(self.focal_kernel_clips[k]// 2) + 2**self.focal_l_clips[k] + (2**self.focal_l_clips[k]-1)
                    kernel_size = self.focal_kernel_clips[k]
                    padding=kernel_size // 2
                    # kernel_size_true=self.focal_kernel_clips[k]+2**self.focal_l_clips[k]-1   
                    kernel_size_true=kernel_size
                relative_position_index_k = getattr(self, 'relative_position_index_clips_{}'.format(k))
                relative_position_bias_to_windows = self.relative_position_bias_table_to_windows_clips[k][:, relative_position_index_k.view(-1)].view(
                    -1, self.window_size[0] * self.window_size[1], (kernel_size_true)**2,
                )
                attn[:, :, :window_area, offset:(offset + (kernel_size_true)**2)] = \
                    attn[:, :, :window_area, offset:(offset + (kernel_size_true)**2)] + relative_position_bias_to_windows.unsqueeze(0)

                # >>>>>>>>>>>>>>>>>>  vanishing point relative position <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                coords_q_moved = coords_q_moving_all[k].view(2, -1)
                coords_q_moved_opposite = coords_q_moving_opposite_all[k].view(2, -1)
                coords_q_moved_sum = list(coords_q_moved[0]*3 + coords_q_moved[1])
                coords_q_moved_opposite_sum = (coords_q_moved_opposite[0]*3 + coords_q_moved_opposite[1])

                relative_position_bias_to_windows_moved_list = [self.relative_position_bias_table_to_windows_clips_vp[k][self.n_poses_mapping[int(i)]].unsqueeze(0) for i in coords_q_moved_sum]
                relative_position_bias_to_windows_moved_opposite_list = [
                    self.relative_position_bias_table_to_windows_clips_vp[k][self.n_poses_mapping[int(i)]].unsqueeze(0) for i in coords_q_moved_opposite_sum]
                relative_position_bias_to_windows_moved = torch.cat(relative_position_bias_to_windows_moved_list, dim=0)
                relative_position_bias_to_windows_moved_opposite = torch.cat(relative_position_bias_to_windows_moved_opposite_list, dim=0)



                attn[:, :, :window_area, (offset + (kernel_size_true)**2):(offset + 2*(kernel_size_true)**2)] = \
                    attn[:, :, :window_area, (offset + (kernel_size_true)**2):(offset + 2*(kernel_size_true)**2)] + relative_position_bias_to_windows_moved
                attn[:, :, :window_area, (offset + 2*(kernel_size_true)**2):(offset + 3*(kernel_size_true)**2)] = \
                    attn[:, :, :window_area, (offset + 2*(kernel_size_true)**2):(offset + 3*(kernel_size_true)**2)] + relative_position_bias_to_windows_moved_opposite

                if mask_all[k+1] is not None:
                    attn[:, :, :window_area, offset:(offset + (kernel_size_true)**2)] = \
                        attn[:, :, :window_area, offset:(offset + (kernel_size_true)**2)] + \
                            mask_all[k+1][:, :, None, None, :].repeat(attn.shape[0] // mask_all[k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+1].shape[-1])
                    vp_mask = mask_all[k + 1][:, mask_coord_list_all[k].type(torch.long), :][0, :, None, None, :]
                    vp_opposite_mask = mask_all[k + 1][:, mask_coord_list_opposite_all[k].type(torch.long), :][0, :, None, None, :]
                    attn[:, :, :window_area, (offset + (kernel_size_true) ** 2):(offset + 2*(kernel_size_true) ** 2)] = \
                        attn[:, :, :window_area, (offset + (kernel_size_true) ** 2):(offset + 2*(kernel_size_true) ** 2)] + vp_mask
                    attn[:, :, :window_area, (offset + 2*(kernel_size_true) ** 2):(offset + 3*(kernel_size_true) ** 2)] = \
                        attn[:, :, :window_area, (offset + 2*(kernel_size_true) ** 2):(offset + 3*(kernel_size_true) ** 2)] + vp_opposite_mask

                offset += (kernel_size_true)**2*3
                # >>>>>>>>>>>>>>>>>>  vanishing point relative position finish<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # print("mask_all[0]: ", mask_all[0])
        # exit()
        if mask_all[0][0] is not None:
            print("inside the mask, be careful 0")
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads, window_area, N)
            attn[:, :, :, :, :window_area] = attn[:, :, :, :, :window_area] + mask_all[0][None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:          
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)

        x = attn @ v_all # (190,8,49,32)
        x = x.transpose(1, 2).reshape(attn.shape[0], window_area, C)  # (190,49,256)

        # >>>>>>>>>>>>>>>>>>>>>>>> Dense VP fusion attention here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        central_window_id_roi = get_central_window_id_roi(
            central_id, self.window_size[0], self.vp_n_windows_ori
        )
        [hmin_window_id, hmax_window_id, wmin_window_id, wmax_window_id] = central_window_id_roi

        nWw = int(nW/self.window_size[1])
        window_ids_vp_grid_h, window_ids_vp_grid_w = torch.meshgrid(
            torch.arange(hmin_window_id, hmax_window_id), torch.arange(wmin_window_id, wmax_window_id)
        )
        window_ids_vp = window_ids_vp_grid_h * nWw + window_ids_vp_grid_w
        window_ids_vp = torch.flatten(window_ids_vp)

        q_windows_ori = q_windows[window_ids_vp, :, :, :].permute(0, 2, 1, 3) # (16, 8, 49, 32)
        q_windows_ori = q_windows_ori.view(B0, self.vp_roi_n_windows_ori*window_area, self.dim) # (16, 8, 49, 32) -> (B, 16*49, 8*32)
        k_windows_vp = k_windows_vp.reshape(B0, self.vp_roi_n_windows*window_area, self.dim) # (B, 64, 49, 256) -> (B, 64*49, 256)
        v_windows_vp = v_windows_vp.reshape(B0, self.vp_roi_n_windows*window_area, self.dim)

        q_windows_ori_attn = self.vanishing_point_attn(
            query=q_windows_ori.permute(1,0,2),
            key=k_windows_vp.permute(1,0,2),
            value=v_windows_vp.permute(1,0,2)
        )[0].permute(1,0,2) # (B, 16*49, 8*32)
        q_windows_ori_attn = self.attn_drop_vp(q_windows_ori_attn)
        q_windows_ori_attn = self.norm_vp(q_windows_ori_attn)
        q_windows_ori_attn = self.proj_vp(q_windows_ori_attn).reshape(-1, window_area, self.dim) # (B, 16*49, 8*32) -> (B*16, 49, 256)

        x[window_ids_vp, :, :] += q_windows_ori_attn

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, window_size, unfold_size):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N        
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CffmTransformerBlock3d3(nn.Module):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        expand_size (int): expand size at first focal level (finest level).
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm 
        pool_method (str): window pooling method. Default: none, options: [none|fc|conv]
        focal_level (int): number of focal levels. Default: 1. 
        focal_window (int): region size of focal attention. Default: 1
        use_layerscale (bool): whether use layer scale for training stability. Default: False
        layerscale_value (float): scaling value for layer scale. Default: 1e-4
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, use_layerscale=False, layerscale_value=1e-4, focal_l_clips=[7,2,4], focal_kernel_clips=[7,5,3]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale
        self.focal_l_clips=focal_l_clips
        self.focal_kernel_clips=focal_kernel_clips

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        self.pool_layers_clips = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1))
                    self.pool_layers[-1].weight.data.fill_(1./(window_size_glo * window_size_glo))
                    self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo, groups=dim))
            for k in range(len(focal_l_clips)):
                # window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if focal_l_clips[k]>self.window_size:
                    window_size_glo = focal_l_clips[k]
                else:
                    window_size_glo = math.floor(self.window_size_glo / (focal_l_clips[k]))
                # window_size_glo = focal_l_clips[k]
                if self.pool_method == "fc":
                    self.pool_layers_clips.append(nn.Linear(window_size_glo * window_size_glo, 1))
                    self.pool_layers_clips[-1].weight.data.fill_(1./(window_size_glo * window_size_glo))
                    self.pool_layers_clips[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers_clips.append(nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo, groups=dim))

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention3d3(
            dim, expand_size=self.expand_size, window_size=to_2tuple(self.window_size), 
            focal_window=focal_window, focal_level=focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pool_method=pool_method, focal_l_clips=focal_l_clips, focal_kernel_clips=focal_kernel_clips)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        print("******self.shift_size: ", self.shift_size)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # B, nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            print("here mask none")
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.vp_n_windows = 8
        self.vp_buffer = 3
        self.vp_stride = 3

        # self.vp_n_windows_clips = [8, 14, 28]
        # self.vp_buffer_clips = [3, 3, 2]
        # self.vp_stride_clips = [3, 2, 1]

    def forward(self, x):
        H0, W0 = self.input_resolution
        # B, L, C = x.shape
        B0, D0, H0, W0, C = x.shape # (1,4,64,128,256)
        shortcut = x
        # assert L == H * W, "input feature has wrong size"
        x=x.reshape(B0*D0,H0,W0,C).reshape(B0*D0,H0*W0,C) # (4, 8192, 256)
        
        x = self.norm1(x)
        x = x.reshape(B0*D0, H0, W0, C) # (4, 64, 128, 256)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W0 % self.window_size) % self.window_size # window_size=7, pad_r=5
        pad_b = (self.window_size - H0 % self.window_size) % self.window_size # window_size=7, pad_b=6
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b)) # (4, 70, 133, 256)
        
        B, H, W, C = x.shape     ## B=B0*D0

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # print("shifted_x.shape: ", shifted_x.shape)
        shifted_x=shifted_x.view(B0,D0,H,W,C) # (1,4,70,133,256)
        x_windows_all = [shifted_x[:,-1]] # (1,70,133,256)
        x_windows_all_clips=[]
        x_window_masks_all = [self.attn_mask] # [None]
        x_window_masks_all_clips=[]
        
        if self.focal_level > 1 and self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none
            # pooling_index=0
            for k in range(self.focal_level-1):     # self.focal_level = 2
                window_size_glo = math.floor(self.window_size_glo / (2 ** k)) # [7]
                pooled_h = math.ceil(H / self.window_size) * (2 ** k) # [10]
                pooled_w = math.ceil(W / self.window_size) * (2 ** k) # [19]
                H_pool = pooled_h * window_size_glo # [70]
                W_pool = pooled_w * window_size_glo # [133]

                x_level_k = shifted_x[:,-1] # [(1,70,133,256),same]
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0,0,0,0,pad_t,pad_b))
                
                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0,0,pad_l,pad_r))

                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), window_size_glo) # B0, nWh, nWw, window_size, window_size, C
                # (1,10,19,7,7,256), nHw = H/window_size = 10

                # vp_n_windows = self.vp_n_windows
                # vanishing_map = vanishing_point_to_depth_mask((H_pool, W_pool)).unsqueeze(0).unsqueeze(0)
                # central_id, [hmin, hmax, wmin, wmax] = get_vanishing_point(
                #     vanishing_map, vp_n_windows, vp_n_windows, window_size_glo,
                #     stride = self.vp_stride, extra_n_windows = self.vp_buffer
                # ) # +3 is for larger margin, providing margin information for unfold
                # vp_roi = [hmin, hmax, wmin, wmax]
                # x_windows_vp_noreshape = window_partition_vanishing_point_no_reshape(
                #     x_level_k.contiguous(), vp_roi, window_size_glo, stride = 3
                # )
                # vp_n_windows_h, vp_n_windows_w = x_windows_vp_noreshape.shape[1], x_windows_vp_noreshape.shape[2]

                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4]) # B0, nWh, nWw, C
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B0, nWh, nWw, C) # B0, nWh, nWw, C                    
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B0, nWh, nWw, window_size_glo*window_size_glo, C).transpose(3, 4) # B0, nWh, nWw, C, wsize**2
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(-2) # B0, nWh, nWw, C

                    # debug: vanishing point
                    # x_windows_vp_noreshape = x_windows_vp_noreshape.view(
                    #     B0, vp_n_windows_h, vp_n_windows_w, window_size_glo*window_size_glo, C
                    # ).transpose(3, 4)
                    # x_windows_vp_pooled = self.pool_layers[k](x_windows_vp_noreshape).flatten(-2) # B0, nWh, nWw, C

                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C).permute(0, 3, 1, 2).contiguous() # B0 * nw * nw, C, wsize, wsize
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B0, nWh, nWw, C) # B0, nWh, nWw, C

                x_windows_all += [x_windows_pooled] # [(1,70,133,256), (1,10,19,256)]
                # print(x_windows_pooled.shape)
                x_window_masks_all += [None]
                # pooling_index=pooling_index+1

            x_windows_all_clips += [x_windows_all]
            x_window_masks_all_clips += [x_window_masks_all]
            for k in range(len(self.focal_l_clips)):
                if self.focal_l_clips[k]>self.window_size:
                    window_size_glo = self.focal_l_clips[k]
                else:
                    window_size_glo = math.floor(self.window_size_glo / (self.focal_l_clips[k])) # [7, 3, 2]
                    pooled_h = math.ceil(H / self.window_size) * (self.focal_l_clips[k])
                    pooled_w = math.ceil(W / self.window_size) * (self.focal_l_clips[k])

                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x[:,k]
                if H!=H_pool or W!=W_pool:
                    x_level_k=F.interpolate(x_level_k.permute(0,3,1,2), size=(H_pool, W_pool), mode='bilinear').permute(0,2,3,1)

                # print(x_level_k.shape)
                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), window_size_glo) # B0, nw, nw, window_size, window_size, C    
                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4]) # B0, nWh, nWw, C
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B0, nWh, nWw, C) # B0, nWh, nWw, C                    
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B0, nWh, nWw, window_size_glo*window_size_glo, C).transpose(3, 4) # B0, nWh, nWw, C, wsize**2
                    x_windows_pooled = self.pool_layers_clips[k](x_windows_noreshape).flatten(-2) # B0, nWh, nWw, C                      
                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C).permute(0, 3, 1, 2).contiguous() # B0 * nw * nw, C, wsize, wsize
                    x_windows_pooled = self.pool_layers_clips[k](x_windows_noreshape).view(B0, nWh, nWw, C) # B0, nWh, nWw, C           

                x_windows_all_clips += [x_windows_pooled]
                # [[(1,70,133,256), (1,10,19,256)], (1,10,19,246), (1,20,38,256), (1,30,57,256)]

                # print(x_windows_pooled.shape)
                x_window_masks_all_clips += [None]
                # pooling_index=pooling_index+1
        # exit()
        
        attn_windows = self.attn(x_windows_all_clips, mask_all=x_window_masks_all_clips, batch_size=B0, num_clips=D0)  # nW*B0, window_size*window_size, C

        attn_windows = attn_windows[:, :self.window_size ** 2] # (190, 49, 245)
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # (190, 7, 7, 256)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (1, 70, 133, 256); B H(padded) W(padded) C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x # (1, 70, 133, 256)
        # x = x[:, :self.input_resolution[0], :self.input_resolution[1]].contiguous().view(B, -1, C)
        x = x[:, :H0, :W0].contiguous().view(B0, -1, C) # (1, 8192, 256)

        # FFN
        # x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        # x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))

        # print(x.shape, shortcut[:,-1].view(B0, -1, C).shape)
        x = shortcut[:,-1].view(B0, -1, C) + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))  # (1, 8192, 256)

        # x=torch.cat([shortcut[:,:-1],x.view(B0,self.input_resolution[0],self.input_resolution[1],C).unsqueeze(1)],1)
        x=torch.cat([shortcut[:,:-1],x.view(B0,H0,W0,C).unsqueeze(1)],1) # (1,3+1,64,128,256)

        assert x.shape==shortcut.shape

        # exit()

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size, self.window_size, self.focal_window)

        if self.pool_method != "none" and self.focal_level > 1:
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                nW_glo = nW * (2**k)
                # (sub)-window pooling
                flops += nW_glo * self.dim * window_size_glo * window_size_glo         
                # qkv for global levels
                # NOTE: in our implementation, we pass the pooled window embedding to qkv embedding layer, 
                # but theoritically, we only need to compute k and v.
                flops += nW_glo * self.dim * 3 * self.dim       

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer3d3(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        expand_size (int): expand size for focal level 1. 
        expand_layer (str): expand layer. Default: all
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm 
        pool_method (str): Window pooling method. Default: none. 
        focal_level (int): Number of focal levels. Default: 1.
        focal_window (int): region size at each focal level. Default: 1. 
        use_conv_embed (bool): whether use overlapped convolutional patch embedding layer. Default: False 
        use_shift (bool): Whether use window shift as in Swin Transformer. Default: False 
        use_pre_norm (bool): Whether use pre-norm before patch embedding projection for stability. Default: False
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False. 
        use_layerscale (bool): Whether use layer scale for stability. Default: False.
        layerscale_value (float): Layerscale value. Default: 1e-4.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer="all",
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, use_conv_embed=False, use_shift=False, use_pre_norm=False, 
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4, focal_l_clips=[16,8,2], focal_kernel_clips=[7,5,3]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1
        
        # build blocks
        self.blocks = nn.ModuleList([
            CffmTransformerBlock3d3(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                 expand_size=0 if (i % 2 == expand_factor) else expand_size, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_level=focal_level, 
                                 focal_window=focal_window, 
                                 use_layerscale=use_layerscale, 
                                 layerscale_value=layerscale_value,
                                 focal_l_clips=focal_l_clips,
                                 focal_kernel_clips=focal_kernel_clips)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x, batch_size=None, num_clips=None):
        B, D, C, H, W = x.shape
        x = rearrange(x, 'b d c h w -> b d h w c')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1, 2).contiguous()
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b d c h w')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
