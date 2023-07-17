# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging

import os
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose

import random
from PIL import Image

@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 invalid_map_suffix=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.invalid_map_suffix = invalid_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)

                        invalid_map = img_name + invalid_map_suffix
                        img_info['ann']['invalid_map'] = invalid_map

                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)

                    invalid_map = img_name + invalid_map_suffix
                    img_info['ann']['invalid_map'] = invalid_map

                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_invalid_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        invalid_maps = []
        for img_info in self.img_infos:
            invalid_map = osp.join(self.ann_dir, img_info['ann']['invalid_map'])
            if not efficient_test and os.path.exists(invalid_map):
                invalid_map = mmcv.imread(
                    invalid_map, flag='unchanged', backend='pillow')
            else:
                return None
            invalid_maps.append(invalid_map)
        return invalid_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        invalid_maps = self.get_invalid_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label,
            invalid_maps=invalid_maps)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

@DATASETS.register_module()
class CustomDataset_cityscape_clips(Dataset):

    CLASSES = None
    PALETTE = None
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 invalid_map_suffix=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=[-9,-6,-3],
                 istraining=True
    ):
        # self.pipeline = Compose(pipeline)
        if istraining:
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.invalid_map_suffix = invalid_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations2(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.invalid_map_suffix, self.split)
        print(len(self.img_infos))
        self.flip_video=True
        print("flip video: ",self.flip_video)
        self.dilation=dilation

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations2(self, img_dir, img_suffix, ann_dir, seg_map_suffix, invalid_map_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)

                        invalid_map = img_name + invalid_map_suffix
                        img_info['ann']['invalid_map'] = invalid_map

                    img_infos.append(img_info)
        else:
            for ann_name in mmcv.scandir(ann_dir, seg_map_suffix, recursive=True):
                img=ann_name.replace(seg_map_suffix, img_suffix)

                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = ann_name
                    img_info['ann'] = dict(seg_map=seg_map)

                    invalid_map = img_name + invalid_map_suffix
                    img_info['ann']['invalid_map'] = invalid_map

                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        if self.flip_video:
            # print("here")
            if random.random()<0.5:
                # imglist=imglist[::-1]
                dilation_used=[-i for i in self.dilation]
            else:
                dilation_used=self.dilation

        # print(img_info, ann_info)
        # exit()
        try:
            img_anns=[]
            for ii in dilation_used:
                img_info_one={}
                filename=img_info['filename']
                seg_map=img_info['ann']['seg_map']
                invalid_map=img_info['ann']['invalid_map']
                value_i_splits=filename.split('_')
                im_name_new = "_".join(
                    value_i_splits[:-2] + [(str(int(value_i_splits[-2]) + ii)).rjust(6, "0")] + value_i_splits[-1:])

                img_info_one['filename']=im_name_new
                img_info_one['ann']=dict(seg_map=seg_map, invalid_map=invalid_map)
                ann_info_one=img_info_one['ann']
                img_anns.append([img_info_one, ann_info_one])
                if not os.path.isfile(self.img_dir+'/'+im_name_new):
                    assert False
        except:
            dilation_used=[-i for i in dilation_used]
            img_anns=[]
            for ii in dilation_used:
                img_info_one={}
                filename=img_info['filename']
                seg_map=img_info['ann']['seg_map']
                invalid_map=img_info['ann']['invalid_map']
                value_i_splits=filename.split('_')
                im_name_new = "_".join(
                    value_i_splits[:-2] + [(str(int(value_i_splits[-2]) + ii)).rjust(6, "0")] + value_i_splits[-1:])

                img_info_one['filename']=im_name_new
                img_info_one['ann']=dict(seg_map=seg_map, invalid_map=invalid_map)
                ann_info_one=img_info_one['ann']
                img_anns.append([img_info_one, ann_info_one])
                if not os.path.isfile(self.img_dir+'/'+im_name_new):
                    assert False
        img_anns.append([img_info, ann_info])

        # print(img_anns)

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        for kkk in img_anns:
            results=dict(img_info=kkk[0], ann_info=kkk[1])
            self.pre_pipeline(results)
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            gt_semantic_seg_clips.append(results["gt_semantic_seg"])

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips,
            vanishing_mask=results["vanishing_mask"],
            pos_emb=results["pos_emb"])

        return self.pipeline_process(results_new)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        img_info = self.img_infos[idx]
        dilation_used=self.dilation
        img_anns=[]
        for ii in dilation_used:
            img_info_one={}
            filename=img_info['filename']
            seg_map=img_info['ann']['seg_map']
            invalid_map=img_info['ann']['invalid_map']
            value_i_splits=filename.split('_')
            im_name_new = "_".join(
                value_i_splits[:-2] + [(str(int(value_i_splits[-2]) + ii)).rjust(6, "0")] + value_i_splits[-1:])
            # value_i_splits=seg_map.split('_')
            # seg_map_new = "_".join(
            #     value_i_splits[:-2] + [(str(int(value_i_splits[-2]) - ii)).rjust(6, "0")] + value_i_splits[-1:])

            img_info_one['filename']=im_name_new
            img_info_one['ann']=dict(seg_map=seg_map, invalid_map=invalid_map)
            ann_info_one=img_info_one['ann']
            img_anns.append([img_info_one, ann_info_one])
            if not os.path.isfile(self.img_dir+'/'+im_name_new):
                assert False
        img_anns.append([img_info, img_info['ann']])

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]
        for kkk in img_anns:
            results = dict(img_info=kkk[0], ann_info=kkk[1])
            self.pre_pipeline(results)
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],
            vanishing_mask=results["vanishing_mask"],
            pos_emb=results["pos_emb"])

        return self.pipeline_process(results_new)
        
    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_invalid_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        invalid_maps = []
        for img_info in self.img_infos:
            invalid_map = osp.join(self.ann_dir, img_info['ann']['invalid_map'])
            if not efficient_test and os.path.exists(invalid_map):
                invalid_map = mmcv.imread(
                    invalid_map, flag='unchanged', backend='pillow')
            else:
                return None
            invalid_maps.append(invalid_map)
        return invalid_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        invalid_maps = self.get_invalid_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label,
            invalid_maps=invalid_maps
        )

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results


    # def evaluate(self,
    #              results,
    #              metric='mIoU',
    #              logger=None,
    #              efficient_test=False,
    #              **kwargs):
    #     """Evaluate the dataset.

    #     Args:
    #         results (list): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
    #             'mDice' are supported.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.

    #     Returns:
    #         dict[str, float]: Default metrics.
    #     """

    #     if isinstance(metric, str):
    #         metric = [metric]
    #     allowed_metrics = ['mIoU', 'mDice']
    #     if not set(metric).issubset(set(allowed_metrics)):
    #         raise KeyError('metric {} is not supported'.format(metric))
    #     eval_results = {}
    #     gt_seg_maps = self.get_gt_seg_maps(efficient_test)
    #     if self.CLASSES is None:
    #         num_classes = len(
    #             reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
    #     else:
    #         num_classes = len(self.CLASSES)
    #     ret_metrics = eval_metrics(
    #         results,
    #         gt_seg_maps,
    #         num_classes,
    #         self.ignore_index,
    #         metric,
    #         label_map=self.label_map,
    #         reduce_zero_label=self.reduce_zero_label)
    #     class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    #     if self.CLASSES is None:
    #         class_names = tuple(range(num_classes))
    #     else:
    #         class_names = self.CLASSES

    #     print("ret_metrics", ret_metrics)

    #     ret_metrics_round = [
    #         np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    #     ]
    #     for i in range(num_classes):
    #         class_table_data.append([class_names[i]] +
    #                                 [m[i] for m in ret_metrics_round[2:]] +
    #                                 [ret_metrics_round[1][i]])
    #     summary_table_data = [['Scope'] +
    #                           ['m' + head
    #                            for head in class_table_data[0][1:]] + ['aAcc']]
    #     ret_metrics_mean = [
    #         np.round(np.nanmean(ret_metric) * 100, 2)
    #         for ret_metric in ret_metrics
    #     ]
    #     summary_table_data.append(['global'] + ret_metrics_mean[2:] +
    #                               [ret_metrics_mean[1]] +
    #                               [ret_metrics_mean[0]])
    #     print_log('per class results:', logger)
    #     table = AsciiTable(class_table_data)
    #     print_log('\n' + table.table, logger=logger)
    #     print_log('Summary:', logger)
    #     table = AsciiTable(summary_table_data)
    #     print_log('\n' + table.table, logger=logger)

    #     for i in range(1, len(summary_table_data[0])):
    #         eval_results[summary_table_data[0]
    #                      [i]] = summary_table_data[1][i] / 100.0
    #     if mmcv.is_list_of(results, str):
    #         for file_name in results:
    #             os.remove(file_name)
    #     return eval_results

