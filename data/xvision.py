'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: xvision.py
@time: 1/1/19 10:16 PM
@desc:
'''
import torch,os
import numpy as np
from utils import print_log
from .data_utils import pil_loader
from sklearn.metrics import auc
from PIL import Image
from PIL import ImageDraw

def evaluate_normalized_mean_error(predictions, groundtruth, log, extra_faces):
    ## compute total average normlized mean error
    assert len(predictions) == len(
        groundtruth), 'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format(
        len(predictions), len(groundtruth))
    assert len(predictions) > 0, 'The length of predictions must be greater than 0 vs {}'.format(len(predictions))
    if extra_faces is not None: assert len(extra_faces) == len(
        predictions), 'The length of extra_faces is not right {} vs {}'.format(len(extra_faces), len(predictions))
    num_images = len(predictions)
    for i in range(num_images):
        c, g = predictions[i], groundtruth[i]
        assert isinstance(c, np.ndarray) and isinstance(g,np.ndarray), 'The type of predictions is not right : [{:}] :: {} vs {} '.format(
            i, type(c), type(g))

    num_points = predictions[0].shape[1]
    error_per_image = np.zeros((num_images, 1))
    for i in range(num_images):
        detected_points = predictions[i]
        ground_truth_points = groundtruth[i]
        if num_points == 68:
            interocular_distance = np.linalg.norm(ground_truth_points[:2, 36] - ground_truth_points[:2, 45])
            assert bool(ground_truth_points[2, 36]) and bool(ground_truth_points[2, 45])
        elif num_points == 51 or num_points == 49:
            interocular_distance = np.linalg.norm(ground_truth_points[:2, 19] - ground_truth_points[:2, 28])
            assert bool(ground_truth_points[2, 19]) and bool(ground_truth_points[2, 28])
        elif num_points == 19:
            assert extra_faces is not None and extra_faces[i] is not None
            interocular_distance = extra_faces[i]
        else:
            raise Exception('----> Unknown number of points : {}'.format(num_points))
        dis_sum, pts_sum = 0, 0
        for j in range(num_points):
            if bool(ground_truth_points[2, j]):
                dis_sum = dis_sum + np.linalg.norm(detected_points[:2, j] - ground_truth_points[:2, j])
                pts_sum = pts_sum + 1
        error_per_image[i] = dis_sum / (pts_sum * interocular_distance)

    normalise_mean_error = error_per_image.mean()
    # calculate the auc for 0.07
    max_threshold = 0.07
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
    area_under_curve07 = auc(threshold, accuracys) / max_threshold
    # calculate the auc for 0.08
    max_threshold = 0.08
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
    area_under_curve08 = auc(threshold, accuracys) / max_threshold

    accuracy_under_007 = np.sum(error_per_image < 0.07) * 100. / error_per_image.size
    accuracy_under_008 = np.sum(error_per_image < 0.08) * 100. / error_per_image.size

    print_log(
        'Compute NME and AUC for {:} images with {:} points :: [(nme): mean={:.3f}, std={:.3f}], auc@0.07={:.3f}, auc@0.08={:.3f}, acc@0.07={:.3f}, acc@0.08={:.3f}'.format(
            num_images, num_points, normalise_mean_error * 100, error_per_image.std() * 100, area_under_curve07 * 100,
            area_under_curve08 * 100, accuracy_under_007, accuracy_under_008), log)

    for_pck_curve = []
    for x in range(0, 3501, 1):
        error_bar = x * 0.0001
        accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
        for_pck_curve.append((error_bar, accuracy))

    return normalise_mean_error, accuracy_under_008, for_pck_curve

class Eval_Meta():

  def __init__(self):
    self.reset()

  def __repr__(self):
    return ('{name}'.format(name=self.__class__.__name__)+'(number of data = {:})'.format(len(self)))

  def reset(self):
    self.predictions = []
    self.groundtruth = []
    self.image_lists = []
    self.face_sizes  = []

  def __len__(self):
    return len(self.image_lists)

  def append(self, _pred, _ground, image_path, face_size):
    assert _pred.shape[0] == 3 and len(_pred.shape) == 2, 'Prediction\'s shape is {:} vs [should be (3,pts) or (2,pts)]'.format(_pred.shape)
    if _ground is not None:
      assert _pred.shape == _ground.shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, _ground.shape)
    if (not self.predictions) == False:
      assert _pred.shape == self.predictions[-1].shape, 'shapes must be the same : {} vs {}'.format(_pred.shape, self.predictions[-1].shape)
    self.predictions.append(_pred)
    self.groundtruth.append(_ground)
    self.image_lists.append(image_path)
    self.face_sizes.append(face_size)

  def save(self, filename):
    meta = {'predictions': self.predictions,
            'groundtruth': self.groundtruth,
            'image_lists': self.image_lists,
            'face_sizes' : self.face_sizes}
    torch.save(meta, filename)
    print ('save eval-meta into {}'.format(filename))

  def load(self, filename):
    assert os.path.isfile(filename), '{} is not a file'.format(filename)
    checkpoint       = torch.load(filename)
    self.predictions = checkpoint['predictions']
    self.groundtruth = checkpoint['groundtruth']
    self.image_lists = checkpoint['image_lists']
    self.face_sizes  = checkpoint['face_sizes']

  def compute_mse(self, log):
    predictions, groundtruth, face_sizes, num = [], [], [], 0
    for x, gt, face in zip(self.predictions, self.groundtruth, self.face_sizes):
      if gt is None: continue
      predictions.append(x)
      groundtruth.append(gt)
      face_sizes.append(face)
      num += 1
    print_log('Filter the unlabeled data from {:} into {:} data'.format(len(self), num), log)
    if num == 0:
      nme, auc, pck_curves = -1, None, None
    else:
      nme, auc, pck_curves = evaluate_normalized_mean_error(self.predictions, self.groundtruth, log, self.face_sizes)
    return nme, auc, pck_curves


def draw_image_by_points(_image, pts, radius, color,facebox, crop, resize):
  if isinstance(_image, str):
    _image = pil_loader(_image)
  assert isinstance(_image, Image.Image), 'image type is not PIL.Image.Image'
  assert isinstance(pts, np.ndarray) and (pts.shape[0] == 2 or pts.shape[0] == 3), 'input points are not correct'
  image, pts = _image.copy(), pts.copy()

  num_points = pts.shape[1]
  visiable_points = []
  for idx in range(num_points):
    if pts.shape[0] == 2 or bool(pts[2,idx]):
      visiable_points.append( True )
    else:
      visiable_points.append( False )
  visiable_points = np.array( visiable_points )
  #print ('visiable points : {}'.format( np.sum(visiable_points) ))


  if crop:
    if isinstance(crop, list):
      x1, y1, x2, y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
    else:
      x1, x2 = pts[0, visiable_points].min(), pts[0, visiable_points].max()
      y1, y2 = pts[1, visiable_points].min(), pts[1, visiable_points].max()
      face_h, face_w = (y2-y1)*0.1, (x2-x1)*0.1
      x1, x2 = int(x1 - face_w), int(x2 + face_w)
      y1, y2 = int(y1 - face_h), int(y2 + face_h)
    image = image.crop((x1, y1, x2, y2))
    pts[0, visiable_points] = pts[0, visiable_points] - x1
    pts[1, visiable_points] = pts[1, visiable_points] - y1
  if resize:
    width, height = image.size
    image = image.resize((resize,resize), Image.BICUBIC)
    pts[0, visiable_points] = pts[0, visiable_points] * 1.0 / width * resize
    pts[1, visiable_points] = pts[1, visiable_points] * 1.0 / height * resize
  finegrain = True
  if finegrain:
    owidth, oheight = image.size
    image = image.resize((owidth*8,oheight*8), Image.BICUBIC)
    pts[0, visiable_points] = pts[0, visiable_points] * 8.0
    pts[1, visiable_points] = pts[1, visiable_points] * 8.0
    radius = radius * 8

  draw  = ImageDraw.Draw(image)
  for idx in range(num_points):
    if visiable_points[ idx ]:
      # draw hollow circle
      point = (pts[0,idx]-radius, pts[1,idx]-radius, pts[0,idx]+radius, pts[1,idx]+radius)
      if radius > 0:
        draw.ellipse(point, fill=color, outline=color)
  if finegrain:
    image = image.resize((owidth,oheight), Image.BICUBIC)
  draw = ImageDraw.Draw(image)
  draw.rectangle(facebox, outline='green')
  return image
