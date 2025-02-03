import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
from torch.nn import init
from dataset import DatasetFromHdf5

from model import *
from running_func import *
from utils import *
import os
from torchinfo import summary

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--model', type=str, default='AHDR')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--format', type=str, default='rgb')
parser.add_argument('--train_data', default='./data_train.txt')
parser.add_argument('--valid_data', default='./data_valid.txt')
parser.add_argument('--test_whole_Image', default='./data_test.txt')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--early_term', type=bool, default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)
parser.add_argument('--batchsize', default=8)
parser.add_argument('--epochs', type=int, default=800000)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--save_model_interval', default=5)

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
train_loaders = torch.utils.data.DataLoader(
    data_loader(args.train_data, patch_div=1, crop_size=256, geometry_aug=True),
    batch_size=args.batchsize, shuffle=True)
valid_loaders = torch.utils.data.DataLoader(
    data_loader(args.valid_data),
    batch_size=1, shuffle=True)

#make folders of trained model and result
if args.run_name:
    trained_model_dir = f"./trained-model-{args.model}-{args.run_name}/"
else:
    trained_model_dir = f"./trained-model-{args.model}/"
trained_model_filename = f"{args.model}_model.pt"
mk_dir(trained_model_dir)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)


##
if args.model in globals():
    model = globals()[args.model]
print(f"\n[INFO] Start training with model {model}")
print(f"\n[INFO] Early termination : {args.early_term}")

model = nn.DataParallel(model(args))
model.apply(weights_init_kaiming)
if args.use_cuda:
    model.cuda()

#summary(
#    model,
#    input_size = [(1, 6, 1000, 1500), (1, 6, 1000, 1500), (1, 6, 1000, 1500)],
#    col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"],
#    verbose=2
#)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
##
start_step = 0

if args.restore and len(os.listdir(trained_model_dir)):
    model, start_step = model_restore(model, trained_model_dir)
    print('restart from {} step'.format(start_step))

early_stopping = EarlyStopping(patience=7, delta=0, mode='min', verbose=True)


start_train = time.time()
for epoch in range(start_step + 1, args.epochs + 1):
    start = time.time()
    train_loss = train(epoch, model, train_loaders, optimizer, trained_model_dir, args)
    end = time.time()
    print('epoch:{}, cost {:.4f} seconds, loss {:.4f}'.format(epoch, end - start, train_loss))
    if epoch % args.save_model_interval == 0:
        model_name = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        torch.save(model.state_dict(), model_name)
        
    valid_loss, psnr, psnr_mu = validation(epoch, model, valid_loaders, trained_model_dir, args)
    early_stopping(train_loss)
    
    fname = trained_model_dir + 'plot_data.txt'
    try: fplot = open(fname, 'a')
    except IOError: print('Cannot open')
    else: 
        fplot.write(f'{epoch},{train_loss},{valid_loss},{psnr},{psnr_mu}\n')
        fplot.close()
    
    if early_stopping.best_update:
        model_name = trained_model_dir + 'best_trained_model_{}.pkl'.format(args.model)
        torch.save(model.state_dict(), model_name)
        print(f"[INFO] Best model saved at epoch {epoch}")
    
    if early_stopping.early_stop and args.early_term:
        print("[INFO] Early Stopped")
        break
end_train = time.time()
print(f"[INFO] Training finished. Total time: {end_train - start_train} seconds")

save_plot(trained_model_dir)
