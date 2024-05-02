python -m torch.distributed.launch --nproc_per_node=4 --master_port=8519 train_PBAFN_e2e.py --name pb_e2e   \
--PBAFN_warp_checkpoint 'checkpoints/pb_s1/PBAFN_warp_epoch_101.pth' --resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 4 --label_nc 14 --launcher pytorch
