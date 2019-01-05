CUDA_VISIBLE_DEVICES=0,1 python ./main.py \
	--train_lists ./datasets/300W_lists/300w.train.DET \
	--eval_ilists ./datasets/300W_lists/300w.test.common.DET \
	              ./datasets/300W_lists/300w.test.challenge.DET \
	              ./datasets/300W_lists/300w.test.full.DET \
        --num_pts 68 \
        --data_indicator 300W-68 \
        --save_path ./snapshots/300W-CPM-DET \
        --eval_once