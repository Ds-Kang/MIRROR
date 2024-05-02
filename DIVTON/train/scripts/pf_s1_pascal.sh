python -m torch.distributed.launch --nproc_per_node=4 --master_port=1519 train_PFAFN_stage1.py --name pf_s1_pascal --parsing pascal  \
--PBAFN_warp_checkpoint 'checkpoints/pb_e2e/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/pb_e2e/PBAFN_gen_epoch_101.pth'  \
--lr 0.00003 --niter 100 --niter_decay 100 --resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 4 --label_nc 14 --launcher pytorch
