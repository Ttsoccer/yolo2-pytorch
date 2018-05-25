import os
import numpy as np
import json

# OpenImage
############################
#path = '/data/unagi0/takayanagi/robocup/OpenImage_for_robocup/class_label.json'
path = './data/open_image_for_robocup/class_label.json'
with open(path, 'r') as f:
    label_names = json.load(f)
num_classes = len(label_names)

# input and output size
############################
multi_scale_inp_size = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        # np.array([608, 608], dtype=np.int),
                        ]   # w, h
multi_scale_out_size = [multi_scale_inp_size[0] / 32,
                        multi_scale_inp_size[1] / 32,
                        multi_scale_inp_size[2] / 32,
                        multi_scale_inp_size[3] / 32,
                        multi_scale_inp_size[4] / 32,
                        multi_scale_inp_size[5] / 32,
                        multi_scale_inp_size[6] / 32,
                        multi_scale_inp_size[7] / 32,
                        multi_scale_inp_size[8] / 32,
                        # multi_scale_inp_size[9] / 32,
                        ]   # w, h
inp_size = np.array([416, 416], dtype=np.int)   # w, h
out_size = inp_size / 32

# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127

base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]

# detection config
############################
object_scale = 5.
noobject_scale = 1.
class_scale = 1.
coord_scale = 1.
iou_thresh = 0.6

# dir config
############################
weight_decay = 0.0005
momentum = 0.9
anchors = np.asarray([(0.57273, 0.677385), (1.87446, 2.06253),
                      (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)],
                      dtype=np.float)
num_anchors = len(anchors)
