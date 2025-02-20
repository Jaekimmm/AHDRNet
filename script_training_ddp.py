import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
from torch.nn import init
from dataset import DatasetFromHdf5

from model import *
from running_func_ddp import *
from utils import *
import os
from torchinfo import summary
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--model', type=str, default='AHDR')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--format', type=str, default='rgb')
parser.add_argument('--early_term', action='store_true', default=False)
parser.add_argument('--train_data', default='./data_train.txt')
parser.add_argument('--valid_data', default='./data_valid.txt')
parser.add_argument('--test_whole_Image', default='./data_test.txt')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)
parser.add_argument('--batchsize', default=8)
parser.add_argument('--epochs', type=int, default=800000)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--save_model_interval', default=10)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.250'
    os.environ['MASTER_PORT'] = '27809'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) 

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def train_ddp(rank, world_size, train_dataset, valid_dataset, trained_model_dir, args):
    #load data
    ddp_setup(rank, world_size)
    print(f"[INFO] (Rank {rank}/{world_size}) started training on CUDA device {torch.cuda.current_device()}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loaders = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, sampler=train_sampler)
    #valid_loaders = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=1, 
    #    sampler=DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False))
    if rank == 0:
        valid_loaders = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    if args.model in globals():
        model = globals()[args.model]
    
    model = model(args).to(rank)
    model.apply(weights_init_kaiming)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    
    if rank == 0:
    
        print(f"\n[INFO] Start training with model {model}")
        print(f"\n[INFO] Early termination : {args.early_term}")
        #early_stopping = EarlyStopping(patience=7, delta=0, mode='min', verbose=True)
    
    start_step = 0
    if args.restore and len(os.listdir(trained_model_dir)):
        model, start_step = model_restore(model, trained_model_dir, rank)
        print(f'[INFO] (Rank {rank}) Restart from {start_step[0]} step')

    for epoch in range(start_step[0] + 1, args.epochs + 1):
        #start = time.time()
        train_sampler.set_epoch(epoch)
        train_loss = train(epoch, model, train_loaders, optimizer, trained_model_dir, args, rank)
        train_loss_tensor = torch.tensor(train_loss, device=rank)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / world_size
        #end = time.time()
        #print('epoch:{}, cost {:.4f} seconds, loss {:.4f}'.format(epoch, end - start, train_loss))
        if rank == 0:
            #early_stopping(train_loss)
            if epoch % args.save_model_interval == 0:
                model_name = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
                torch.save(model.state_dict(), model_name)
                valid_loss, psnr, psnr_mu = validation(epoch, model, valid_loaders, trained_model_dir, args)

                fname = trained_model_dir + 'plot_data.txt'
                try: fplot = open(fname, 'a')
                except IOError: print('Cannot open')
                else: 
                    fplot.write(f'{epoch},{train_loss},{valid_loss},{psnr},{psnr_mu}\n')
                    fplot.close()

            #if early_stopping.best_update:
            #    model_name = trained_model_dir + 'best_trained_model_{}.pkl'.format(args.model)
            #    torch.save(model.state_dict(), model_name)
            #    print(f"[INFO] Best model saved at epoch {epoch}")

        #if early_stopping.early_stop and args.early_term:
        #    print("[INFO] Early Stopped")
        #    break
    dist.destroy_process_group()

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    print("\n\n << CUDA devices >>")
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        cuda_count = torch.cuda.device_count()
        print(f"Number of visible CUDA devices: {cuda_count}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
    else:
        print("CUDA is not available.\n")
    
    #make folders of trained model and result
    if args.run_name:
        trained_model_dir = f"./trained-model-{args.model}-{args.run_name}/"
    else:
        trained_model_dir = f"./trained-model-{args.model}/"
    trained_model_filename = f"{args.model}_model.pt"
    mk_dir(trained_model_dir)
    
    train_dataset = data_loader(args.train_data, patch_div=1, crop_size=256, geometry_aug=True)
    print(f"[INFO] Train dataset loaded: {len(train_dataset)} images")
    valid_dataset = data_loader(args.valid_data)
    
    start_train = time.time()
    
    mp.spawn(train_ddp, args=(cuda_count, train_dataset, valid_dataset, trained_model_dir, args), nprocs=cuda_count, join=True)
    save_plot(trained_model_dir)

    end_train = time.time()
    print(f"[INFO] Training finished. Total time: {end_train - start_train} seconds")