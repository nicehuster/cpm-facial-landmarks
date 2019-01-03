# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
CUDA_VISIBLE_DEVICES=0,1 python ./main.py \
	--train_lists ./datasets/AFLW_lists/train.GTB \
	--eval_ilists ./datasets/AFLW_lists/test.GTB \
	--num_pts 19 \
        --data_indicator AFLW-19 \
	--save_path ./snapshots/AFLW-CPM-DET 
	
