'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: demo.py
@time: 1/1/19 10:07 PM
@desc:
'''

from __future__ import division
import torch,argparse,os,cv2
from pathlib import Path
import numpy as np

from data import transforms, GeneralDataset,draw_image_by_points
from utils import load_configure
from models import obtain_model,remove_module_dict
os.environ['CUDA_VISIBLE_DEVICES']='1'
def face_detect(path,face_detector):
    image=cv2.imread(path)
    haar_face_cascade = cv2.CascadeClassifier(face_detector)
    gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    # only return one face information
    assert len(faces)>0,'No face found.'
    return [faces[0][0],faces[0][1],faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]]

def evaluate(args):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print('The image is {:}'.format(args.image))
    print('The model is {:}'.format(args.model))
    snapshot = Path(args.model)
    assert snapshot.exists(), 'The model path {:} does not exist'
    facebox=face_detect(args.image,args.face_detector)

    print('The face bounding box is {:}'.format(facebox))
    assert len(facebox)==4,'Invalid face input : {:}'.format(facebox)
    snapshot = torch.load(str(snapshot))

    # General Data Argumentation
    mean_fill = tuple([int(x * 255) for x in [0.485, 0.456, 0.406]])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    param = snapshot['args']
    eval_transform = transforms.Compose(
        [transforms.PreCrop(param.pre_crop_expand), transforms.TrainScale2WH((param.crop_width, param.crop_height)),
         transforms.ToTensor(), normalize])
    model_config = load_configure(param.model_config, None)
    dataset = GeneralDataset(eval_transform, param.sigma, model_config.downsample, param.heatmap_type, param.data_indicator)
    dataset.reset(param.num_pts)

    net = obtain_model(model_config, param.num_pts + 1)
    net = net.cuda()
    weights = remove_module_dict(snapshot['state_dict'])
    net.load_state_dict(weights)
    print('Prepare input data')
    [image, _, _, _, _, _, cropped_size], meta = dataset.prepare_input(args.image, facebox)
    inputs = image.unsqueeze(0).cuda()
    # network forward
    with torch.no_grad():
        batch_heatmaps, batch_locs, batch_scos = net(inputs)
    # obtain the locations on the image in the orignial size
    cpu = torch.device('cpu')
    np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(
        cpu).numpy(), cropped_size.numpy()
    locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)

    scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2), cropped_size[1] * 1. / inputs.size(-1)

    locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + \
                                       cropped_size[3]
    prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)

    print('the coordinates for {:} facial landmarks:'.format(param.num_pts))
    for i in range(param.num_pts):
        point = prediction[:, i]
        print('the {:02d}/{:02d}-th point : ({:.1f}, {:.1f}), score = {:.2f}'.format(i+1, param.num_pts, float(point[0]),
                                                                                     float(point[1]), float(point[2])))
    image = draw_image_by_points(args.image, prediction, 2, (255, 0, 0), facebox, None,None)
    image.show()
    image.save(args.image.split('.')[0]+'_result.jpg')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate a single image by the trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image',            type=str,   help='The evaluation image path.')
  parser.add_argument('--model',            type=str,   help='The snapshot to the saved detector.')
  parser.add_argument('--face_detector',    default='haarcascade_frontalface_alt2.xml',type=str,   help='The face detector')
  args = parser.parse_args()
  evaluate(args)
