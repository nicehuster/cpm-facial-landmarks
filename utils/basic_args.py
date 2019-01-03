'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: basic_args.py
@time: 1/1/19 7:06 PM
@desc:
'''

import os,sys,time,random,argparse

def obtain_args():
    parser=argparse.ArgumentParser(description='Train facial landmark detectors on 300-W or AFLW',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_lists', type=str, nargs='+', help='The list file path to the video training dataset.')
    parser.add_argument('--eval_ilists', type=str, nargs='+', help='The list file path to the image testing dataset.')
    parser.add_argument('--num_pts', type=int, help='Number of point.')
    parser.add_argument('--model_config', default='./configs/Detector.config',type=str, help='The path to the model configuration')
    parser.add_argument('--opt_config', type=str, default='./configs/SGD.config',help='The path to the optimizer configuration')
    # Data Generation
    parser.add_argument('--heatmap_type', type=str, default='gaussian',choices=['gaussian', 'laplacian'],
                        help='The method for generating the heatmap.')
    parser.add_argument('--data_indicator', type=str, help='The method for generating the heatmap.')
    # Data Transform
    parser.add_argument('--pre_crop_expand', type=float, default=0.2,help='parameters for pre-crop expand ratio')
    parser.add_argument('--sigma', type=float, default=4,help='sigma distance for CPM.')
    parser.add_argument('--scale_prob', type=float, default=1.0,help='argument scale probability.')
    parser.add_argument('--scale_min', type=float, default=0.9,help='argument scale : minimum scale factor.')
    parser.add_argument('--scale_max', type=float, default=1.1,help='argument scale : maximum scale factor.')
    parser.add_argument('--scale_eval', type=float, default=1,help='argument scale : maximum scale factor.')
    parser.add_argument('--rotate_max', type=int, default=20,help='argument rotate : maximum rotate degree.')
    parser.add_argument('--crop_height', type=int, default=256, help='argument crop : crop height.')
    parser.add_argument('--crop_width', type=int, default=256, help='argument crop : crop width.')
    parser.add_argument('--crop_perturb_max', type=int,default=30, help='argument crop : center of maximum perturb distance.')
    parser.add_argument('--arg_flip', default=False,action='store_true', help='Using flip data argumentation or not ')
    # Optimization options
    parser.add_argument('--eval_once', default=False,action='store_true', help='evaluation only once for evaluation ')
    parser.add_argument('--error_bar', type=float, help='For drawing the image with large distance error.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    # Checkpoints
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency (default: 200)')
    parser.add_argument('--save_path', type=str,  help='Folder to save checkpoints and log.')
    # Acceleration
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 2)')
    # Random Seed
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    args = parser.parse_args()
    if args.rand_seed is None:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_path is not None, 'save-path argument can not be None'

    return args

if __name__=='__main__':
    args = obtain_args()
    for name, value in args._get_kwargs():
        print('{:16} : {:}'.format(name, value))
