import time
import argparse
import torch.utils.data

from model import *
from running_func_ch import *
from utils import *

import os

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--model', type=str, default='AHDR')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--format', type=str, default='rgb')
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--test_whole_Image', default='./data_test.txt')
parser.add_argument('--use_cuda', default=True)
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
    cuda_count = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {cuda_count}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
else:
    print("CUDA is not available.\n")
    
#load data
testimage_dataset = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image, color=args.format),
    batch_size=1)
valid_loaders = torch.utils.data.DataLoader(
    data_loader(args.test_whole_Image, color=args.format),
    batch_size=1, shuffle=True)

#make folders of trained model and result
if args.run_name:
    if args.epoch > 0:
        outdir = f"./result-{args.model}-{args.run_name}-e{args.epoch}/"
    else:
        outdir = f"./result-{args.model}-{args.run_name}-best/"
else:
    if args.epoch > 0:
        outdir = f"./result-{args.model}-e{args.epoch}/"
    else:
        outdir = f"./result-{args.model}-best/"
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
    
#model = model(args)
model = nn.DataParallel(model(args))
model.apply(weights_init_kaiming)
if args.use_cuda:
    model.cuda()

##
start_step = 0
# if args.load_model and len(os.listdir(args.trained_model_dir)):
if args.run_name:
    trained_model_dir = f"./trained-model-{args.model}-{args.run_name}/"
else:
    trained_model_dir = f"./trained-model-{args.model}/"
#trained_model_filename = f"{args.model}_model.pt"
if args.epoch > 0:
    trained_model_filename = f"trained_model{args.epoch}.pkl"
else:
    #trained_model_filename = f"{args.model}_model.pt"
    trained_model_filename = f"best_trained_model_{args.model}.pkl"
model = model_load(model, trained_model_dir, trained_model_filename)

# In the testing, we test on the whole image, so we defind a new variable
#  'Image_test_loaders' used to load the whole image
start = time.time()
loss = testing_fun(model, testimage_dataset, outdir, args)

end = time.time()
print('Running Time: {} seconds'.format(end - start))
