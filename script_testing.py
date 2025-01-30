import time
import argparse
import torch.utils.data

from model import *
from running_func import *
from utils import *

import os

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--model', type=str, default='AHDR',)
parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--cuda_devices', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()


torch.manual_seed(args.seed)
print("\n\n << CUDA devices >>")
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    if args.cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    print(f"Number of visible CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
else:
    print("CUDA is not available.\n")
    
#load data
testimage_dataset = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image),
    batch_size=1)

#make folders of trained model and result
outdir = f"./result_{args.model}"
mk_dir(outdir)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

##
if args.model in globals():
    model = globals()[args.model]
print(f"\nRun test with model {model}\n")

model = model(args)
model.apply(weights_init_kaiming)
if args.use_cuda:
    model.cuda()

##
start_step = 0
# if args.load_model and len(os.listdir(args.trained_model_dir)):
trained_model_dir = f"./trained-model-{args.model}"
trained_model_filename = f"{args.model}_model.pt"
model = model_load(model, trained_model_dir, trained_model_filename)

# In the testing, we test on the whole image, so we defind a new variable
#  'Image_test_loaders' used to load the whole image
start = time.time()
loss = testing_fun(model, testimage_dataset, args)
end = time.time()
print('Running Time: {} seconds'.format(end - start))
