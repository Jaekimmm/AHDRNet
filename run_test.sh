#python script_testing.py --model "AHDR" --run_name "org"
#python script_testing.py --model "AHDR" --epoch "250"
#python script_testing.py --model "AHDR" --epoch "500"
#python script_testing.py --model "AHDR" --epoch "750"
#python script_testing.py --model "AHDR" --epoch "1000"
#python script_testing.py --model "AHDR" --run_name "mono_f64" --format "mono" --nChannel 2 --nFeat 64 --epoch "250"
#python script_testing.py --model "AHDR" --run_name "mono_f64" --format "mono" --nChannel 2 --nFeat 64 --epoch "500"
#python script_testing.py --model "AHDR" --run_name "mono_f64" --format "mono" --nChannel 2 --nFeat 64 --epoch "750"
#python script_testing.py --model "AHDR" --run_name "mono_f64" --format "mono" --nChannel 2 --nFeat 64 --epoch "1000"
#python script_testing.py --model "AHDR" --run_name "mono_f16" --format "mono" --nChannel 2 --nFeat 16 --epoch "250"
#python script_testing.py --model "AHDR" --run_name "mono_f16" --format "mono" --nChannel 2 --nFeat 16 --epoch "500"
#python script_testing.py --model "AHDR" --run_name "mono_f16" --format "mono" --nChannel 2 --nFeat 16 --epoch "750"
#python script_testing.py --model "AHDR" --run_name "mono_f16" --format "mono" --nChannel 2 --nFeat 16 --epoch "1000"
#python script_testing.py --model "AHDR" --run_name "ddp_test" --epoch "1100"
#python script_testing.py --model "AHDR" --run_name "yuv444" --format "444" --epoch "250"
#python script_testing.py --model "AHDR" --run_name "yuv444" --format "444" --epoch "500"
#python script_testing.py --model "AHDR" --run_name "yuv444" --format "444" --epoch "750"
#python script_testing.py --model "AHDR" --run_name "yuv444" --format "444" --epoch "1000"
#python script_testing_ch.py --model "AHDR" --run_name "chRGB" --format "rgb" --nChannel 2 --epoch "250"
#python script_testing.py --model "LIGHTFUSE" --epoch 40 --format "rgb_dual" --nChannel 6 --nFeat 32
#python script_testing.py --model "LIGHTFUSE_sigmoid" --run_name 'vgg' --epoch 100 --format "rgb_dual_sice" --test_whole_Image "./dataset_sice/test" --nChannel 6 --nFeat 32
#CUDA_VISIBLE_DEVICES=3 python script_testing.py --model "LIGHTFUSE" --run_name 'sice' --epoch 10 --format "rgb_dual_sice" --test_whole_Image "./dataset_sice/valid" --nChannel 6 --nFeat 32 --offset
#CUDA_VISIBLE_DEVICES=3 python script_testing.py --model "LIGHTFUSE_sigmoid" --run_name 'sice' --epoch 10 --format "rgb_dual_sice" --test_whole_Image "./dataset_sice/valid" --nChannel 6 --nFeat 32
#CUDA_VISIBLE_DEVICES=3 python script_testing.py --model "LIGHTFUSE_bilinear_upscale" --run_name 'sice' --epoch 10 --format "rgb_dual_sice" --test_whole_Image "./dataset_sice/valid" --nChannel 6 --nFeat 32 --offset
#CUDA_VISIBLE_DEVICES=3 python script_testing.py --model "LIGHTFUSE" --run_name 'kalan' --epoch 20 --format "rgb_org" --test_whole_Image "./dataset_test" --nChannel 6 --nFeat 32 --offset
#CUDA_VISIBLE_DEVICES=1 python script_testing.py --model "LIGHTFUSE" --run_name 'kalan_tm' --epoch 20 --format "rgb_tm" --test_whole_Image "./dataset_test" --nChannel 6 --nFeat 32 --offset
#CUDA_VISIBLE_DEVICES=3 python script_testing.py --model "LIGHTFUSE_sigmoid_skip_short" --run_name 'sice' --epoch 10 --format "rgb_dual_sice" --test_whole_Image "./dataset_sice/valid" --nChannel 6 --nFeat 32
CUDA_VISIBLE_DEVICES=3 python script_testing.py --model "LIGHTFUSE_sigmoid" --run_name 'kalan-mix' --epoch 20 --format "rgb_tm" --test_whole_Image "./dataset_test" --nChannel 6 --nFeat 32 --label_tonemap
