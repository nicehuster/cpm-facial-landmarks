# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp
from PIL import Image
from scipy.ndimage.interpolation import zoom
import numpy as np
import copy, math

def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    with Image.open(f) as img:
      return img.convert('RGB')

def PTSconvert2str(points):
  assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
  assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
  string = ''
  num_pts = points.shape[1]
  for i in range(num_pts):
    ok = False
    if points.shape[0] == 3 and bool(points[2, i]) == True:
      ok = True
    elif points.shape[0] == 2:
      ok = True

    if ok:
      string = string + '{:02d} {:.2f} {:.2f} True\n'.format(i+1, points[0, i], points[1, i])
  string = string[:-1]
  return string

def PTSconvert2box(points, expand_ratio=None):
  assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
  assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
  if points.shape[0] == 3:
    points = points[:2, points[-1,:].astype('bool') ]
  elif points.shape[0] == 2:
    points = points[:2, :]
  else:
    raise Exception('The shape of points is not right : {}'.format(points.shape))
  assert points.shape[1] >= 2, 'To get the box of points, there should be at least 2 vs {}'.format(points.shape)
  box = np.array([ points[0,:].min(), points[1,:].min(), points[0,:].max(), points[1,:].max() ])
  W = box[2] - box[0]
  H = box[3] - box[1]
  assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)
  if expand_ratio is not None:
    box[0] = int( math.floor(box[0] - W * expand_ratio) )
    box[1] = int( math.floor(box[1] - H * expand_ratio) )
    box[2] = int( math.ceil(box[2] + W * expand_ratio) )
    box[3] = int( math.ceil(box[3] + H * expand_ratio) )
  return box

def for_generate_box_str(anno_path, num_pts, extend):
  if isinstance(anno_path, str):
    points, _ = anno_parser(anno_path, num_pts)
  else:
    points = anno_path.copy()
  box = PTSconvert2box(points, extend)
  return '{:.2f} {:.2f} {:.2f} {:.2f}'.format(box[0], box[1], box[2], box[3])

def resize_heatmap(maps, height, width, order=3):
  # maps  = np.ndarray with shape [height, width, channels]
  # order = 0 Nearest
  # order = 1 Bilinear
  # order = 2 Cubic
  assert isinstance(maps, np.ndarray) and len(maps.shape) == 3, 'maps type : {}'.format(type(maps))

  scale = tuple(np.array([height,width], dtype=float) / np.array(maps.shape[:2]))
  return zoom(maps, scale + (1,), order=order)

def analysis_dataset(dataset):
  all_values = np.zeros((3,len(dataset.datas)), dtype=np.float64)
  hs = np.zeros((len(dataset.datas),), dtype=np.float64)
  ws = np.zeros((len(dataset.datas),), dtype=np.float64)

  for index, image_path in enumerate(dataset.datas):
    img = pil_loader(image_path)
    ws[index] = img.size[0]
    hs[index] = img.size[1]
    img = np.array(img)
    all_values[:, index] = np.mean(np.mean(img, axis=0), axis=0).astype('float64')
  mean = np.mean(all_values, axis=1)
  std  = np.std (all_values, axis=1)
  return mean, std, ws, hs

def split_datasets(dataset, point_ids):
  sub_dataset = copy.deepcopy(dataset)
  assert len(point_ids) > 0
  assert False, 'un finished'

def convert68to49(points):
  points = points.copy()
  assert len(points.shape) == 2 and (points.shape[0] == 3 or points.shape[0] == 2) and points.shape[1] == 68, 'The shape of points is not right : {}'.format(points.shape)
  out = np.ones((68,)).astype('bool')
  out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,60,64]] = False
  cpoints = points[:, out]
  assert len(cpoints.shape) == 2 and cpoints.shape[1] == 49
  return cpoints

def convert68to51(points):
  points = points.copy()
  assert len(points.shape) == 2 and (points.shape[0] == 3 or points.shape[0] == 2) and points.shape[1] == 68, 'The shape of points is not right : {}'.format(points.shape)
  out = np.ones((68,)).astype('bool')
  out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]] = False
  cpoints = points[:, out]
  assert len(cpoints.shape) == 2 and cpoints.shape[1] == 51
  return cpoints

