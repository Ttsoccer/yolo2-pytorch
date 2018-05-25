import os
import numpy as np
import json
import cv2
import time
import scipy.sparse
from PIL import Image
import sys
import _pickle as pickle

import threading

sys.path.append('../')
from pycocotools.coco import COCO
from .imdb import ImageDataset
from torch.utils.data import Dataset # new point 

class COCODataset(ImageDataset, Dataset):
    def __init__(self, data_type, year, datadir, batch_size, im_processor, cfg, processes=5, shuffle=True, dst_size=None):
        self.imdb_name = '%s%s'%(data_type, year)
        ImageDataset.__init__(self,'coco_'+self.imdb_name, datadir, batch_size, im_processor, cfg, processes, shuffle, dst_size)
        Dataset.__init__(self)
        anno_path = os.path.join(datadir, 'data', 'annotations', 'instances_%s%s.json'%(data_type, year))
        self.coco = COCO(annotation_file=anno_path)
        self.year = str(year)
        self._load_class_ids()
        self._image_ids = self._get_image_ids()
        print('load annotations and image_names')
        st = time.time()
        self._annotations,self._image_names = self._load_annotation()
        print('done, time=%5.2f'%(time.time()-st))
        self._image__indexes = np.arange(len(self._image_names))

    def __len__(self):
        return len(self._image__indexes)

    def __getitem__(self, index):
        return index

    def _get_image_ids(self):
        return self.coco.getImgIds()

    def _get_image_path(self, index):
        image_path = os.path.join(self._data_dir, 'data', self.year, self.imdb_name, 'COCO_%s_%012d.jpg'%(self.imdb_name, index))
        return image_path

    def _load_class_ids(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in categories])
        self.class_to_ind = dict(zip(self.classes, np.arange(self.num_classes)))
        self.class_to_coco_cat_ind = dict(zip([c['name'] for c in categories], self.coco.getCatIds()))
        self.coco_cat_id_to_class_ind = dict([(self.class_to_coco_cat_ind[cls], self.class_to_ind[cls]) for cls in self.classes[1:]])

    def _load_annotation(self):
        cache_file = os.path.join(self.cache_path, self.imdb_name + '_gt_roidb_image_names.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb, image_names = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.imdb_name, cache_file))
            return roidb, image_names

        gt_roidb = []
        image_names = []
        for image_id in self._image_ids:
            roidb = self._load_annotations_from_index(image_id)
            image_path = self._get_image_path(image_id)

            if os.path.exists(image_path):
                gt_roidb.append(roidb)
                image_names.append(image_path)
        print(image_path)
        with open(cache_file, 'wb') as fid:
            pickle.dump((gt_roidb,image_names), fid)
        print('wrote gt roidb to {}'.format(cache_file))
        print(image_names[:5])
        return gt_roidb, image_names

    def _load_annotations_from_index(self, index):
        im_ann = self.coco.loadImgs([index])[0]
        width = im_ann['width']
        height = im_ann['height']
        ann_ids = self.coco.getAnnIds(imgIds=[index], iscrowd=None)
        objs = self.coco.loadAnns(ann_ids)

        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        num_objs = len(valid_objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for i, obj in enumerate(valid_objs):
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            boxes[i, :] = obj['clean_bbox']
            gt_classes[i] = cls
            seg_areas[i] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[i, :] = -1.0
            else:
                overlaps[i, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

    ### new point ###
    def fetch_betch_data(self, ith, x, size_index):
        images, gt_boxes, classes, dontcare, origin_im = self._im_processor(
            [self.image_names[x], self.get_annotation(x), self.dst_size], size_index=None, data_name=self.name)
        
        # multi-scale
        w, h = self.cfg.multi_scale_inp_size[size_index]
        gt_boxes = np.asarray(gt_boxes, dtype=np.float)
        if len(gt_boxes) > 0:
            gt_boxes[:, 0::2] *= float(w) / images.shape[1]
            gt_boxes[:, 1::2] *= float(h) / images.shape[0]
        images = cv2.resize(images, (w, h))
        
        self.batch['images'][ith] = images
        self.batch['gt_boxes'][ith] = gt_boxes
        self.batch['gt_classes'][ith] = classes
        self.batch['dontcare'][ith] = dontcare
        self.batch['origin_im'][ith] = origin_im
        
    def parse(self, index, size_index):
        index = index.numpy()
        lenindex = len(index)
        self.batch = {'images': [list()] * lenindex,
                      'gt_boxes': [list()] * lenindex,
                      'gt_classes': [list()] * lenindex,
                      'dontcare': [list()] * lenindex,
                      'origin_im': [list()] * lenindex}
        ths = []
        for ith in range(lenindex):
            #print('each ith:',ith)
            ths.append(threading.Thread(target=self.fetch_betch_data, args=(ith, index[ith], size_index)))
            ths[ith].start()
        for ith in range(lenindex):
            ths[ith].join()
        self.batch['images'] = np.asarray(self.batch['images'])

        return self.batch

if __name__=="__main__":
    import sys
    sys.path.append('../utils')
    import utils.yolo as yolo_utils
    im_processor = yolo_utils.preprocess_train
    data_type = 'val'
    year = str(2014)
    imdb = COCODataset(data_type, year, datadir=root_path, batch_size=10, im_processor=im_processor)
    print(imdb._annotations[0])
    print(imdb.image_names[0])
    print(os.path.exists(imdb.image_names[0]))
