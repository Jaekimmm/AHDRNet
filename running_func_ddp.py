import os
import random
import numpy as np
import torch
import h5py
import time

import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import glob
from utils import *
import imageio
from datetime import datetime

def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def model_restore(model, trained_model_dir, rank):
    if rank == 0:
        model_list = glob.glob((trained_model_dir + "/trained_*.pkl"))
        a = []
        for i in range(len(model_list)):
            index = int(model_list[i].split('model')[-1].split('.')[0])
            a.append(index)
        epoch = np.sort(a)[-1]
        model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        #model.load_state_dict(torch.load(model_path))
        print(f'[INFO] Load model from {model_path}')
    else:
        model_path = None
        epoch = 0
        
    model_path = [model_path]
    epoch = [epoch]
    dist.broadcast_object_list(model_path, src=0)
    dist.broadcast_object_list(epoch, src=0)
    
    if model_path[0] is not None:
        model.load_state_dict(torch.load(model_path[0], map_location=f'cuda:{rank}'))
    
    return model, epoch


class data_loader(data.Dataset):
    def __init__(self, list_dir, patch_div=1, crop_size=0, geometry_aug=False):
        super().__init__()
        self.patch_div = patch_div
        self.crop_size = crop_size
        self.geometry_aug = geometry_aug
        
        with open(list_dir) as f:
            self.list_txt = f.readlines()
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

    def __getitem__(self, index):
        sample_path = self.list_txt[index]
        sample_path = sample_path.strip()

        if os.path.exists(sample_path):

            f = h5py.File(sample_path, 'r')
            #data = f['IN'][:]
            #label = f['GT'][:]
            data = self.crop_for_patch(f['IN'][:], self.patch_div)
            label = self.crop_for_patch(f['GT'][:], self.patch_div)
            f.close()
            #crop_size = 256
            if self.crop_size > 0 : data, label = self.imageCrop(data, label, self.crop_size)
            if self.geometry_aug : data, label = self.image_Geometry_Aug(data, label)


        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, label, crop_size):
        c, w, h = data.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...

        start_w = self.random_number(w_boder - 1)
        start_h = self.random_number(h_boder - 1)

        crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        return crop_data, crop_label

    def image_Geometry_Aug(self, data, label):
        c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:
            in_data = data
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, index]
            in_label = in_label[:, :, index]

        return in_data, in_label
    
    def crop_for_patch(self, image, patch_div=1):
        # crop the image to be divisible by patch_div
        if image.ndim == 2:  # 2D image (height x width)
            height, width = image.shape
            new_height = (height // patch_div) * patch_div
            new_width = (width // patch_div) * patch_div
            cropped_image = image[:new_height, :new_width]
        elif image.ndim == 3:  # 3D image (channels x height x width)
            channels, height, width = image.shape
            new_height = (height // patch_div) * patch_div
            new_width = (width // patch_div) * patch_div
            cropped_image = image[:, :new_height, :new_width]
        elif image.ndim == 4:  # 4D image (num x channels x height x width)
            num, channels, height, width = image.shape
            new_height = (height // patch_div) * patch_div
            new_width = (width // patch_div) * patch_div
            cropped_image = image[:, :, :new_height, :new_width]
        else:
            raise ValueError("Unsupported image shape")
    
        return cropped_image
        

def get_lr(epoch, lr, max_epochs):
    #if epoch <= max_epochs * 0.8:
    #    lr = lr
    #else:
    #    lr = 0.1 * lr
    return lr

def train(epoch, model, train_loaders, optimizer, trained_model_dir, args, rank):
    lr = get_lr(epoch, args.lr, args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if rank == 0: print('[INFO] lr: {}'.format(optimizer.param_groups[0]['lr']))
    
    model.train()
    num = 0
    trainloss = 0
    avg_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loaders):
        if args.use_cuda:
            data, target = data.to(rank), target.to(rank)

############  used for End-to-End code
        if args.format == 'mono':
            data_mono = torch.cat([rgb_to_mono(data[:, 3*i:3*(i+1), :, :]) for i in range(6)], dim=1)  # (batch, 6, 1, H, W)
            data1 = torch.cat((data_mono[:, 0:1, :, :], data_mono[:, 3:4, :, :]), dim=1)
            data2 = torch.cat((data_mono[:, 1:2, :, :], data_mono[:, 4:5, :, :]), dim=1)
            data3 = torch.cat((data_mono[:, 2:3, :, :], data_mono[:, 5:6, :, :]), dim=1)
        else:
            data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
            data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
            data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)
        
        optimizer.zero_grad()
        output = model(data1, data2, data3)

#########  make the loss
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        
        trainloss += loss.item()
        avg_loss += loss.item()
        
        print(f'train (rank {rank}) Epoch {epoch} iteration: {batch_idx}, loss: {loss.item():.6f}')

    avg_loss /= len(train_loaders)
    
    return avg_loss

def testing_fun(model, test_loaders, outdir, args):
    model.eval()
    test_loss = 0
    val_psnr = 0
    val_psnr_mu = 0
    num = 0
    
    for data, target in test_loaders:
        Test_Data_name = test_loaders.dataset.list_txt[num].split('.h5')[0].split('/')[-1]
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            if args.format == 'mono':
                data_mono = torch.cat([rgb_to_mono_gt(data[:, 3*i:3*(i+1), :]) for i in range(6)], dim=1)  # (batch, 6, 1, H, W)
                data1 = torch.cat((data_mono[:, 0:1, :], data_mono[:, 3:4, :]), dim=1)
                data2 = torch.cat((data_mono[:, 1:2, :], data_mono[:, 4:5, :]), dim=1)
                data3 = torch.cat((data_mono[:, 2:3, :], data_mono[:, 5:6, :]), dim=1)
            else:
                data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
                data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
                data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
            output = model(data1, data2, data3)

        # save the result to .H5 files
        hdrfile = h5py.File(outdir + "/" + Test_Data_name + '_hdr.h5', 'w')
        img = output[0, :, :, :]
        img = tv.utils.make_grid(img.data.cpu()).numpy()
        hdrfile.create_dataset('data', data=img)
        hdrfile.close()
        
        # save the result as tif w/ tonemapping (copy from freeSoul)
        gamma = 2.24 #degamma
        img = torch.squeeze(output)
        img = img.data.cpu().numpy().astype(np.float32)
        img = np.transpose(img, (2, 1, 0))
        img = img[:, :, [0, 1, 2]]
        img = img ** gamma
        norm_perc = np.percentile(img, 99)
        img = tanh_norm_mu_tonemap(img, norm_perc)
        imageio.imwrite(outdir + "/" + Test_Data_name + '.tif', img, 'tif')
        
        #########  Prepare to calculate metrics
        psnr_output = torch.squeeze(output[0].clone())
        psnr_target = torch.squeeze(target.clone())
        psnr_output = psnr_output.data.cpu().numpy().astype(np.float32)
        psnr_target = psnr_target.data.cpu().numpy().astype(np.float32)
        
        #########  Calculate metrics
        psnr = normalized_psnr(psnr_output, psnr_target, psnr_target.max())
        psnr_mu = psnr_tanh_norm_mu_tonemap(psnr_target, psnr_output)

        val_psnr += psnr
        val_psnr_mu += psnr_mu
        
        hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
            Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(
            Variable(torch.from_numpy(np.array([1 + 5000])).float()))

        test_loss += F.mse_loss(hdr, target)
        num = num + 1

    test_loss = test_loss / len(test_loaders.dataset)
    val_psnr = val_psnr / len(test_loaders.dataset)
    val_psnr_mu = val_psnr_mu / len(test_loaders.dataset)
    print('\n Test set: Average Loss: {:.4f}'.format(test_loss.item()))

    run_time = datetime.now().strftime('%m/%d %H:%M:%S')
    flog = open('./test_result.log', 'a')
    flog.write(f'{args.model}, {args.run_name}, {arhs.epoch}, {run_time}, {test_loss:.6f}, {val_psnr:.6f}, {val_psnr_mu:.06f}\n')
    flog.close()
    return test_loss


def validation(epoch, model, valid_loaders, trained_model_dir, args):
    model.eval()
    val_psnr = 0
    val_psnr_mu = 0
    valid_loss = 0
    valid_num = len(valid_loaders)
    
    for batch_idx, (data, target) in enumerate(valid_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            if args.format == 'mono':
                data_mono = torch.cat([rgb_to_mono_gt(data[:, 3*i:3*(i+1), :]) for i in range(6)], dim=1)  # (batch, 6, 1, H, W)
                data1 = torch.cat((data_mono[:, 0:1, :], data_mono[:, 3:4, :]), dim=1)
                data2 = torch.cat((data_mono[:, 1:2, :], data_mono[:, 4:5, :]), dim=1)
                data3 = torch.cat((data_mono[:, 2:3, :], data_mono[:, 5:6, :]), dim=1)
            else:
                data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
                data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
                data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
            output = model(data1, data2, data3)
    
        #########  make the loss
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        
        loss = F.l1_loss(output, target)
        valid_loss = valid_loss + loss
        
        #########  Prepare to calculate metrics
        psnr_output = torch.squeeze(output[0].clone())
        psnr_target = torch.squeeze(target.clone())
        psnr_output = psnr_output.data.cpu().numpy().astype(np.float32)
        psnr_target = psnr_target.data.cpu().numpy().astype(np.float32)
        
        #########  Calculate metrics
        psnr = normalized_psnr(psnr_output, psnr_target, psnr_target.max())
        psnr_mu = psnr_tanh_norm_mu_tonemap(psnr_target, psnr_output)

        val_psnr = val_psnr + psnr
        val_psnr_mu = val_psnr_mu + psnr_mu
        
    valid_loss = valid_loss / valid_num
    val_psnr = val_psnr / valid_num
    val_psnr_mu = val_psnr_mu / valid_num
    print('Validation Epoch {}: avg_loss: {:.4f}, Average PSNR: {:.4f}, PSNR_mu: {:.4f}'.format(epoch, valid_loss, val_psnr, val_psnr_mu))
    
    fname = trained_model_dir + '/psnr.txt'
    try:
        fobj = open(fname, 'a')
    except IOError:
        print('open error')
    else:
        fobj.write('Epoch {}: Average PSNR: {:.4f}, PSNR_mu: {:.4f}\n'.format(epoch, val_psnr, val_psnr_mu))
        fobj.close()
    
    return valid_loss, val_psnr, val_psnr_mu

class testimage_dataloader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

    def __getitem__(self, index):
        sample_path = self.list_txt[index]
        sample_path = sample_path.strip()
        
        if os.path.exists(sample_path):
            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)