python -m torch.distributed.launch --nproc_per_node=4 --master_port=9519 train_PFAFN_e2e.py --name pf_e2e_pascal  --parsing pascal \
--PFAFN_warp_checkpoint 'checkpoints/pf_s1_pascal/PFAFN_warp_epoch_201.pth'  \
--PBAFN_warp_checkpoint 'checkpoints/pb_e2e/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/pb_e2e/PBAFN_gen_epoch_101.pth'  \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 4 --label_nc 14 --launcher pytorch
