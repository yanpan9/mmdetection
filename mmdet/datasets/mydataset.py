import os 
import re 
import json 
import logging
import base64
import os.path as osp
import tempfile

from io import BytesIO
from PIL import Image, ImageOps

import numpy as np
from pycocotools.cocoeval import COCOeval

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class MyDataset(CustomDataset):

    CLASSES = ("spot", "rough", "half", "cross")

    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i+1 for i, cat in enumerate(self.CLASSES)}

    def load_annotations(self, ann_file):
        ''' 
        Args:
            ann_file (str): For my dataset, it's a path which stores the jsons
        '''
        img_infos = list()
        files = os.listdir(ann_file)
        for ele in files:
            file_path = osp.join(ann_file, ele)
            with open(file_path, "r") as src:
                data = json.load(src)
                info = dict()
                img = self.base64_to_image(data["imageData"])
                image = np.asarray(img)
                if img.mode == "RGB":
                    image = image[:,:,::-1]
                info["filename"] = image
                info["jsonname"] = ele
                info["height"] = img.height
                info["width"] = img.width
                info["shapes"] = data["shapes"]
                img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_info = self.img_infos[idx]
        ann_info = img_info["shapes"]
        return self._parse_ann_info(ann_info)

    def _parse_ann_info(self, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            x0 = min(ann["points"], key=lambda x:x[0])[0]
            y0 = min(ann["points"], key=lambda x:x[1])[1]
            x1 = max(ann["points"], key=lambda x:x[0])[0]
            y1 = max(ann["points"], key=lambda x:x[1])[1]
            bbox = [x0, y0, x1, y1]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann["label"]])
            seg = [list(np.asarray(ann["points"]).flatten())]
            gt_masks_ann.append(seg)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann)

        return ann

    def base64_to_image(self, base64_str, image_path=None):
        base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
        byte_data = base64.b64decode(base64_data)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        if image_path:
            img.save(image_path)
        return img


if __name__ == "__main__":
    pass