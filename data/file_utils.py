'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: file_utils.py
@time: 1/1/19 10:13 PM
@desc:
'''
import glob
import numpy as np
def load_txt_file(file_path):
  '''
  load data or string from text file.
  '''
  with open(file_path, 'r') as cfile:
    content = cfile.readlines()
  cfile.close()
  content = [x.strip() for x in content]
  num_lines = len(content)
  return content, num_lines
def load_list_from_folder(folder_path, ext_filter=None, depth=1):
  '''
  load a list of files or folders from a system path

  parameter:
    folder_path: root to search
    ext_filter: a string to represent the extension of files interested
    depth: maximum depth of folder to search, when it's None, all levels of folders will be searched
  '''
  folder_path = osp.normpath(folder_path)
  assert isinstance(depth, int) , 'input depth is not correct {}'.format(depth)
  assert ext_filter is None or (isinstance(ext_filter, list) and all(isinstance(ext_tmp, str) for ext_tmp in ext_filter)) or isinstance(ext_filter, str), 'extension filter is not correct'
  if isinstance(ext_filter, str):    # convert to a list
    ext_filter = [ext_filter]

  fulllist = list()
  wildcard_prefix = '*'
  for index in range(depth):
    if ext_filter is not None:
      for ext_tmp in ext_filter:
        curpath = osp.join(folder_path, wildcard_prefix + '.' + ext_tmp)
        fulllist += glob.glob(curpath)
    else:
      curpath = osp.join(folder_path, wildcard_prefix)
      fulllist += glob.glob(curpath)
    wildcard_prefix = osp.join(wildcard_prefix, '*')

  fulllist = [osp.normpath(path_tmp) for path_tmp in fulllist]
  num_elem = len(fulllist)

  return fulllist, num_elem
def load_list_from_folders(folder_path_list, ext_filter=None, depth=1):
  '''
  load a list of files or folders from a list of system path
  '''
  assert isinstance(folder_path_list, list) or isinstance(folder_path_list, str), 'input path list is not correct'
  if isinstance(folder_path_list, str):
    folder_path_list = [folder_path_list]

  fulllist = list()
  num_elem = 0
  for folder_path_tmp in folder_path_list:
    fulllist_tmp, num_elem_tmp = load_list_from_folder(folder_path_tmp, ext_filter=ext_filter, depth=depth)
    fulllist += fulllist_tmp
    num_elem += num_elem_tmp

  return fulllist, num_elem

def remove_item_from_list(list_to_remove, item):
  '''
  remove a single item from a list
  '''
  assert isinstance(list_to_remove, list), 'input list is not a list'

  try:
    list_to_remove.remove(item)
  except ValueError:
    print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

  return list_to_remove

def anno_parser(anno_path, num_pts):
  data, num_lines = load_txt_file(anno_path)
  if data[0].find('version: ') == 0: # 300-W
    return anno_parser_v0(anno_path, num_pts)
  else:
    return anno_parser_v1(anno_path, num_pts)

def anno_parser_v0(anno_path, num_pts):
  '''
  parse the annotation for 300W dataset, which has a fixed format for .pts file
  return:
    pts: 3 x num_pts (x, y, oculusion)
  '''
  data, num_lines = load_txt_file(anno_path)
  assert data[0].find('version: ') == 0, 'version is not correct'
  assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'
  assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'

  assert data[0] == 'version: 1' or data[0] == 'version: 1.0', 'The version is wrong : {}'.format(data[0])
  n_points = int(data[1][len('n_points: '):])

  assert num_lines == n_points + 4, 'number of lines is not correct'    # 4 lines for general information: version, n_points, start and end symbol
  assert num_pts == n_points, 'number of points is not correct'

  # read points coordinate
  pts = np.zeros((3, n_points), dtype='float32')
  line_offset = 3    # first point starts at fourth line
  point_set = set()
  for point_index in range(n_points):
    try:
      pts_list = data[point_index + line_offset].split(' ')       # x y format
      if len(pts_list) > 2:    # handle edge case where additional whitespace exists after point coordinates
        pts_list = remove_item_from_list(pts_list, '')
      pts[0, point_index] = float(pts_list[0])
      pts[1, point_index] = float(pts_list[1])
      pts[2, point_index] = float(1)      # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by 300-W
      point_set.add( point_index )
    except ValueError:
      print('error in loading points in %s' % anno_path)
  return pts, point_set

def anno_parser_v1(anno_path, NUM_PTS, one_base=True):
  '''
  parse the annotation for MUGSY-Full-Face dataset, which has a fixed format for .pts file
  return: pts: 3 x num_pts (x, y, oculusion)
  '''
  data, n_points = load_txt_file(anno_path)
  assert n_points <= NUM_PTS, '{} has {} points'.format(anno_path, n_points)
  # read points coordinate
  pts = np.zeros((3, NUM_PTS), dtype='float32')
  point_set = set()
  for line in data:
    try:
      idx, point_x, point_y, oculusion = line.split(' ')
      idx, point_x, point_y, oculusion = int(idx), float(point_x), float(point_y), oculusion == 'True'
      if one_base==False: idx = idx+1
      assert idx >= 1 and idx <= NUM_PTS, 'Wrong idx of points : {:02d}-th in {:s}'.format(idx, anno_path)
      pts[0, idx-1] = point_x
      pts[1, idx-1] = point_y
      pts[2, idx-1] = float( oculusion )
      point_set.add(idx)
    except ValueError:
      raise Exception('error in loading points in {}'.format(anno_path))
  return pts, point_set

def merge_lists_from_file(file_paths, seed=None):
  assert file_paths is not None, 'The input can not be None'
  if isinstance(file_paths, str):
    file_paths = [ file_paths ]
  print ('merge lists from {} files with seed={} for random shuffle'.format(len(file_paths), seed))
  # load the data
  all_data = []
  for file_path in file_paths:
    assert osp.isfile(file_path), '{} does not exist'.format(file_path)
    listfile = open(file_path, 'r')
    listdata = listfile.read().splitlines()
    listfile.close()
    all_data = all_data + listdata
  total = len(all_data)
  print ('merge all the lists done, total : {}'.format(total))
  # random shuffle
  if seed is not None:
    np.random.seed(seed)
    order = np.random.permutation(total).tolist()
    new_data = [ all_data[idx] for idx in order ]
    all_data = new_data
  return all_data


from os import path as osp

def load_file_lists(file_paths):
  if isinstance(file_paths, str):
    file_paths = [ file_paths ]
  print ('Function [load_lists] input {:} files'.format(len(file_paths)))
  all_strings = []
  for file_idx, file_path in enumerate(file_paths):
    assert osp.isfile(file_path), 'The {:}-th path : {:} is not a file.'.format(file_idx, file_path)
    listfile = open(file_path, 'r')
    listdata = listfile.read().splitlines()
    listfile.close()
    print ('Load [{:d}/{:d}]-th list : {:} with {:} images'.format(file_idx, len(file_paths), file_path, len(listdata)))
    all_strings += listdata
  return all_strings