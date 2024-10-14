srun python distill_quant.py --cfg /home/uqbyuan3/uqbyuan3/data-distillation/autopalette/configs/CIFAR-10/ConvIN/IPC10.yaml \
    --lr_quant 0.05 --num_colors 64 \
    --conf_ratio 0.0 \
    --color_model color_cnn \
    --subset_ckpt /home/uqbyuan3/uqbyuan3/data-distillation/DATM/selection/result/CIFAR10_Medcut_ConvNetD3_GraphCut_64_exp0_epoch_40_ \
    # --info_ratio 0.0 \
    # --colormax_ratio 0.0 \
    # --subset_ckpt /home/uqbyuan3/uqbyuan3/normal/DeepCore/result/CIFAR10_ConvNet_Submodular_graphcut_ipc10.ckpt  \
    # --diverse_loss --diverse_loss_ratio 0.05 --diverse_type simple_mmd \
    # --info_ratio 0.0 \
    # --background_mask --patch_ratio 0.6 \
    # --colormax_ratio 0.0 \


srun python DATM_quant.py --cfg /clusterdata/uqbyuan3/data_distillation/DATM/configs/CIFAR-100/ConvIN/IPC44.yaml \
    --lr_quant 0.05 --num_colors 64 \
    --conf_ratio 0.0 \
    --color_model color_cnn \
    --subset_ckpt /clusterdata/uqbyuan3/data_distillation/DATM/selection/result/CIFAR100_Medcut_ConvNetD3_GraphCut_64_exp0_epoch_44_


python DATM_quant.py --cfg /clusterdata/uqbyuan3/data_distillation/DATM/configs/TinyImageNet/ConvIN/IPC4.yaml \
    --lr_quant 0.05 --num_colors 64 \
    --conf_ratio 0.0 \
    --color_model color_cnn \
    --subset_ckpt /clusterdata/uqbyuan3/data_distillation/DATM/selection/result/CIFAR100_Medcut_ConvNetD3_GraphCut_64_exp0_epoch_44_ \
    --use_warmup 



sun python dm_quant.py --cfg /clusterdata/uqbyuan3/data_distillation/DATM/configs/DM/CIFAR-100/ConvIN/IPC200.yaml \
    --lr_quant 0.05 --num_colors 64 \
    --conf_ratio 0.0 \
    --color_model color_cnn \
    --subset_ckpt /clusterdata/uqbyuan3/data_distillation/DATM/selection/result/CIFAR100_Medcut_ConvNetD3_GraphCut_64_exp0_epoch_200_



python DATM_quant.py --cfg /clusterdata/uqbyuan3/data_distillation/DATM/configs/ImageNet/fruit/IPC40.yaml \
    --lr_quant 0.05 --num_colors 64 \
    --conf_ratio 0.0 \
    --color_model color_cnn \
    --pixsim_ratio 10.0 \
    --colormax_ratio 3.0 \
    --info_ratio 3.0 \
    --use_warmup



salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=20G --job-name=TinyInteractive --time=01:00:00 --partition=gpu_cuda_debug --gres=gpu:h100:1  --account=a_huang srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l


salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=20G --job-name=TinyInteractive --time=10:00:00 --partition=gpu_cuda --gres=gpu:h100:1 --account=a_huang srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
