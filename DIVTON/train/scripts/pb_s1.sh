python -m torch.distributed.launch --nproc_per_node=4 --master_port=7119 train_PBAFN_stage1.py --name pb_s1   \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 4 --label_nc 14 --launcher pytorch
