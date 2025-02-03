#CUDA_VISIBLE_DEVICES=4,5,6,7 \
python script_training.py --model 'AHDR' --format 'rgb' --early_term True
#python script_training.py --model AHDR_mono --format 'mono' --nChannel 2 --nFeat 16 --epochs 100 --early_term 0
#python script_training.py --model AHDR_mono_f64 --format 'mono' --nChannel 2 --nFeat 64 --epochs 100 --early_term 0
#python script_training.py --model SOL_2_6_2_2 --epochs 100 --early_term True
