#CUDA_VISIBLE_DEVICES=3,5,6,7 python script_training.py --model 'AHDR' --run_name 'yuv422' --format '422' --epochs 1000
#CUDA_VISIBLE_DEVICES=3,5,6,7 python script_training.py --model 'AHDR' --run_name 'yuv444' --format '444' --epochs 250
#CUDA_VISIBLE_DEVICES=0,1,2,3 python script_training_ddp.py --model 'AHDR' --run_name 'ddp_test' --format 'rgb' --epochs 1100
#CUDA_VISIBLE_DEVICES=0,1,2,3 python script_training.py --model 'AHDR' --run_name 'mono_f64' --format 'mono' --nChannel 2 --nFeat 64 --epochs 1000
#python script_training.py --model 'AHDR' --run_name 'mono_f16' --format 'mono' --nChannel 2 --nFeat 16 --epochs 1000
#python script_training.py --model 'AHDR' --format 'rgb' --early_term
#CUDA_VISIBLE_DEVICES=3,5,6,7 python script_training_ch.py --model 'AHDR' --run_name 'chRGB' --format 'rgb' --nChannel 2 --epochs 250
#CUDA_VISIBLE_DEVICES=0,1,2,3 python script_training.py --model 'LIGHTFUSE' --run_name 'vgg_loss' --format 'rgb_dual' --nChannel 3 --nFeat 32 --epochs 1000 --lr 0.001
#CUDA_VISIBLE_DEVICES=0,1,2,3 python script_training.py --model 'LIGHTFUSE' --run_name 'dual-vgg_loss' --format 'rgb_dual' --nChannel 6 --nFeat 32 --epochs 1000 --lr 0.001
#CUDA_VISIBLE_DEVICES=0 python script_training.py --model 'LIGHTFUSE' --run_name 'no_offset' --loss 'vgg' --format 'rgb_dual' --nChannel 6 --nFeat 32 --epochs 100 --lr 0.001
#CUDA_VISIBLE_DEVICES=2 python script_training.py --model 'LIGHTFUSE_sigmoid' --run_name 'vgg' --loss 'vgg' --format 'rgb_dual' --nChannel 6 --nFeat 32 --epochs 100 --lr 0.001
#CUDA_VISIBLE_DEVICES=6,7 python script_training.py \
#    --model 'LIGHTFUSE' \
#    --run_name 'sice' \
#    --loss 'vgg' \
#    --format 'rgb_dual_sice' \
#    --train_data './dataset_sice/train' \
#    --valid_data './dataset_sice/test' \
#    --offset \
#    --nChannel 6 \
#    --nFeat 32 \
#    --epochs 100 \
#    --batchsize 128 \
#    --lr 0.001 \
#    --save_model_interval 10
CUDA_VISIBLE_DEVICES=6,7 python script_training.py \
    --model 'LIGHTFUSE_sigmoid' \
    --run_name 'kalan-mix-labeltm' \
    --loss 'vgg' \
    --format 'rgb_tm' \
    --train_data './dataset_train_patch' \
    --valid_data './dataset_test' \
    --nChannel 6 \
    --nFeat 32 \
    --epochs 20 \
    --batchsize 8 \
    --lr 0.001 \
    --save_model_interval 10 \
    --label_tonemap
#CUDA_VISIBLE_DEVICES=2,3 python script_training.py \
#    --model 'LIGHTFUSE_sigmoid_skip_long' \
#    --run_name 'sice' \
#    --loss 'vgg' \
#    --format 'rgb_dual_sice' \
#    --train_data './dataset_sice/train' \
#    --valid_data './dataset_sice/valid' \
#    --nChannel 6 \
#    --nFeat 32 \
#    --epochs 10 \
#    --batchsize 128 \
#    --lr 0.001 \
#    --save_model_interval 1
#