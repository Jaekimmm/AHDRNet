CUDA_VISIBLE_DEVICES=0,1,2,3 python script_training.py --model 'AHDR' --run_name 'mono_f64' --format 'mono' --nChannel 2 --nFeat 64 --epochs 1000
#python script_training.py --model 'AHDR' --run_name 'mono_f16' --format 'mono' --nChannel 2 --nFeat 16 --epochs 1000
#python script_training.py --model 'AHDR' --format 'rgb' --early_term
#python script_training.py --model 'AHDR' --epochs 1000 --format 'rgb'
#python script_training.py --model AHDR --format 'mono' --nChannel 2 --nFeat 64 --epochs 100 --early_term 0 \
#python script_training.py --model SOL_2_6_2_2 --epochs 100 --early_term True