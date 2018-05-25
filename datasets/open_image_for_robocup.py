import os
import numpy as np
import json
import cv2
import scipy.sparse
import _pickle as pickle
import time

import threading

from .imdb import ImageDataset
from torch.utils.data import Dataset # new point 

class OpenImageForRobocupDataset(ImageDataset, Dataset):
    def __init__(self, data_type, datadir, batch_size, im_processor, cfg, processes=5, shuffle=True, dst_size=None):
        self.imdb_name = 'OpenImageForRobocup_%s'%(data_type)
        self.data_type = data_type
        ImageDataset.__init__(self, self.imdb_name, datadir, batch_size, im_processor, cfg, processes, shuffle, dst_size)
        Dataset.__init__(self)
        anno_path = os.path.join(datadir, '%s_annotation.json'%(data_type))
        self._classes = cfg.label_names
        self._annotations,self._image_names = self._load_annotation(anno_path)
        self._image__indexes = np.arange(len(self._image_names))
        
    def __len__(self):
        return len(self._image__indexes)

    def __getitem__(self, index):
        return index

    def _get_annotation_dict(self, anno_path):
        st= time.time()
        print('load annotaiton dict')
        with open(anno_path, 'r') as f:
            anno_dict = json.load(f)
        print('done, time:%.2f'%(time.time()-st))
        return anno_dict

    def _get_image_path(self, index):
        image_path = os.path.join(self._data_dir, 'raw_data', self.data_type, '%s.jpg'%(index))
        return image_path

    def _load_annotation(self, anno_path):
        print('load annotations and image_names')
        st = time.time()
        cache_file = os.path.join(self.cache_path, self.imdb_name + '_gt_roidb_image_names.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb, image_names = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.imdb_name, cache_file))
            print('time : %.2f'%(time.time()-st))
            return roidb, image_names

        anno_dicts = self._get_annotation_dict(anno_path)
        self._image_ids = list(anno_dicts.keys())
        gt_roidb = []
        image_names = []
        for i, image_id in enumerate(self._image_ids):
            if (i+1) %100000==0:
                print('i : %d, time : %.2f'%(i+1, time.time()-st))
            roidb = self._load_annotations_from_index(anno_dicts[image_id])
            image_path = self._get_image_path(image_id)
            if os.path.exists(image_path):
                gt_roidb.append(roidb)
                image_names.append(image_path)
        with open(cache_file, 'wb') as fid:
            pickle.dump((gt_roidb,image_names), fid)
        print('wrote gt roidb to {}'.format(cache_file))
        print('time : %.2f'%(time.time()-st))

        return gt_roidb, image_names

    def _load_annotations_from_index(self, anno_dict):
        boxes = np.array(anno_dict['boxes'], dtype=np.float32)
        if len(boxes)==0:
            boxes = np.empty([0,4], dtype=np.float32)
        gt_classes = np.array(anno_dict['class_ids'], dtype=np.int32)
        num_objs = boxes.shape[0]
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        for i, gt_class in enumerate(gt_classes):
            overlaps[i, gt_class] = 1.0
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
    data_type = 'train'
    datadir = '/data/unagi0/takayanagi/robocup/OpenImage/'
    batch_size = 10
    im_processor = yolo_utils.preprocess_train
    processes = 2
    shuffle = True
    dst_size = cfg.multi_scale_inp_size
    imdb = OpenImageDataset(data_type, datadir, batch_size=batch_size, im_processor=im_processor, cfg=cfg)
    print(imdb._annotations[0])
    print(imdb.image_names[0])
    print(os.path.exists(imdb.image_names[0]))
