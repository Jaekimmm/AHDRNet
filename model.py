import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

        
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Attention Guided HDR, AHDR-Net
class AHDR(nn.Module):
    def __init__(self, args):
        super(AHDR, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nDenselayer : {nDenselayer}, nFeat : {nFeat}, growthRate : {growthRate}")

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)

        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()


    def forward(self, x1, x2, x3):

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = torch.sigmoid(F1_A)
        F1_ = F1_ * F1_A


        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = torch.sigmoid(F3_A)
        F3_ = F3_ * F3_A

        F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        us = self.conv_up(FDF)

        output = self.conv3(us)
        output = torch.sigmoid(output)

        return output

class SOL_2_6_2_2(nn.Module):
    def __init__(self, args):
        super(SOL_2_6_2_2, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        self.gamma = 0.45


        ### Extract Feature Map

        self.conv1 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )

        self.conv3 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )
        
        self.conv1_2 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )
        
        self.conv2_2 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )
        
        self.conv2_3 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )
        
        self.conv3_3 = nn.Sequential(
          nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1),
          nn.ReLU()
        )
        
        
        
        #############

        self.att1 = nn.Sequential(
          nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True),
          nn.ReLU(),
          nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True),
          nn.Sigmoid()
        )
        
        self.att3 = nn.Sequential(
          nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True),
          nn.ReLU(),
          nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True),
          nn.Sigmoid()
        )
        
        #############
        ## Align
        #############

        self.align_mul1 = nn.Sequential(
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
          nn.ReLU()
        )
        
        self.align_add1 = nn.Sequential(
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
          nn.ReLU()
        )

        self.align_mul3 = nn.Sequential(
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
          nn.ReLU()
        )
        
        self.align_add3 = nn.Sequential(
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
          nn.ReLU()
        )

        
        self.convOut1 = nn.Sequential(
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1),
          nn.ReLU(),
          # nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1),
          # nn.Sigmoid()
        )

        self.convOut2 = nn.Sequential(
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1),
          nn.ReLU(),
          # nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1),
          # nn.Sigmoid()
        )

        self.convOut3 = nn.Sequential(
          nn.Conv2d(nFeat*3, nFeat*3, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(nFeat*3, nFeat*3, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(nFeat*3, nFeat*3, kernel_size=3, padding=1),
          nn.ReLU(),
        )
        
        self.convOut4 = nn.Sequential(
          nn.Conv2d(nFeat*3, nChannel, kernel_size=1, padding=0),
          nn.Sigmoid()
        )
        
    def forward(self, x1, x2, x3, x1_att_in_1, x2_att_in_1, x2_att_in_3, x3_att_in_3):
      
            # data1 = torch.cat((data[:, 18:21, :, :], data[:, 9:12, :, :]), dim=1) LDR(normalized) + HDR_short
            # data2_1 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1) LDR(mid) + HDR_ref
            # data2_2 = torch.cat((data[:, 21:24, :, :], data[:, 12:15, :, :]), dim=1) LDR(mid_long normlized) + HDR_ref
            # data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1) LDR + HDR_long
        
        # x1_sat = self.imageThresClipping(x1, 1/exp[:,1], 'upper')
        # x1_att_in_1 = self.imageApplyGainAndClip(x1_sat, exp[:,1], self.gamma)
        # x2_att_in_1 = self.imageApplyGainAndClip(x2, exp[:,1], self.gamma)
        
        # x2_sat = self.imageThresClipping(x2, 1/exp[:,2], 'upper')
        # x2_att_in_3 = self.imageApplyGainAndClip(x2_sat, exp[:,2], self.gamma)
        # x3_att_in_3 = self.imageApplyGainAndClip(x3, exp[:,2], self.gamma)
        
        # low HDR + saturated image
              

        F1_ = self.conv1(x1)
        F2_ = self.conv2(x2)
        F3_ = self.conv3(x3)
        
        F1A_2 = self.conv1_2(x1_att_in_1)
        F2A_2 = self.conv2_2(x2_att_in_1)
        F2A_3 = self.conv2_3(x2_att_in_3)
        F3A_3 = self.conv3_3(x3_att_in_3)
        
        F12_ = torch.cat((F1A_2, F2A_2), 1)
        F23_ = torch.cat((F2A_3, F3A_3), 1)
        
        F1_att = self.att1(F12_)
        F3_att = self.att3(F23_)
        
        F1_ = F1_ * F1_att
        F3_ = F3_ * F3_att
        
        F1_MUL = self.align_mul1(F1_)
        F1_ADD = self.align_add1(F1_)
        
        F3_MUL = self.align_mul3(F3_)
        F3_ADD = self.align_add3(F3_)
        
        F12_Aligned = F1_MUL * F2_ + F1_ADD
        
        F12_Aligned = self.convOut1(F12_Aligned)
        
        F23_Aligned = F3_MUL * F2_ + F3_ADD
        
        F23_Aligned = self.convOut2(F23_Aligned)
        
        output = torch.cat((F12_Aligned, F23_Aligned, F2_), 1)  
        
        output = self.convOut3(output)
        
        output = self.convOut4(output)
        
        return output      
 
class LIGHTFUSE(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        #self.upsample1 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
        #self.upsample2 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
        #self.upsample3 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
        #TODO:test with bi-linear
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        #out = torch.sigmoid(out)
        out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] tanh_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_sigmoid(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_bilinear_upscale(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_bilinear_upscale, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        #out = torch.sigmoid(out)
        out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] tanh_out: {out.min()} ~ {out.max()}")
        return out

class LIGHTFUSE_sigmoid_skip_long(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid_skip_long, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out + x3  # skip long exposure
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 
class LIGHTFUSE_sigmoid_skip_short(nn.Module):
    def __init__(self, args):
        super(LIGHTFUSE_sigmoid_skip_short, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args
        print(f"[INFO] nChannel : {nChannel}, nFeat : {nFeat}")
        
        # DetailNet
        self.pointwise1 = nn.Conv2d(nChannel, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise2 = nn.Conv2d(nFeat, nFeat, kernel_size=1, stride=1, padding=0, bias=True)
        self.pointwise3 = nn.Conv2d(nFeat, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # GlobalNet : 
        # Depthwise & Separable Convolution Layers
        self.depthwise1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.depthwise2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel)
        self.separable_conv = nn.Sequential(
          nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=2, padding=1, groups=nChannel),
          nn.Conv2d(nChannel,        3, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Upsampling (ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)

        # Final Output
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        DEBUG_FLAG = 0
        
        if DEBUG_FLAG == 2: print(f"[INFO] x1: {x1.min():.6f} ~ {x1.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x2: {x2.min():.6f} ~ {x2.max():.6f}")
        if DEBUG_FLAG == 2: print(f"[INFO] x3: {x3.min():.6f} ~ {x3.max():.6f}")
        
        x = torch.cat((x1, x3), dim=1)
        
        # DetailNet
        if DEBUG_FLAG == 1: print(f"[INFO] x : {x.shape}")
        dnet_1 = self.relu(self.pointwise1(x))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_1 : {dnet_1.shape}")
        dnet_2 = self.relu(self.pointwise2(dnet_1))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_2 : {dnet_2.shape}")
        dnet_out = self.relu(self.pointwise3(dnet_2))
        if DEBUG_FLAG == 1: print(f"[INFO] dnet_out : {dnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] dnet_out: {dnet_out.min()} ~ {dnet_out.max()}")
        
        # GlobalNet 
        gnet_dconv_1 = self.depthwise1(x)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_1 : {gnet_dconv_1.shape}")
        gnet_dconv_2 = self.depthwise2(gnet_dconv_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_2 : {gnet_dconv_2.shape}")
        gnet_dconv_3 = self.separable_conv(gnet_dconv_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_dconv_3 : {gnet_dconv_3.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_dconv_3: {gnet_dconv_3.min()} ~ {gnet_dconv_3.max()}")

        # Upsampling
        gnet_up_1 = self.upsample1(gnet_dconv_3)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_1 : {gnet_up_1.shape}")
        gnet_up_2 = self.upsample2(gnet_up_1)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_2 : {gnet_up_2.shape}")
        gnet_out = self.upsample3(gnet_up_2)
        if DEBUG_FLAG == 1: print(f"[INFO] gnet_up_3 : {gnet_out.shape}")
        if DEBUG_FLAG == 2: print(f"[INFO] gnet_out: {gnet_out.min()} ~ {gnet_out.max()}")

        # Element-wise Addition
        out = dnet_out + gnet_out + x1  # skip short exposure
        if DEBUG_FLAG == 2: print(f"[INFO] add_out: {out.min()} ~ {out.max()}")
        out = torch.sigmoid(out)
        #out = self.tanh(out)
        if DEBUG_FLAG == 2: print(f"[INFO] sigmoid_out: {out.min()} ~ {out.max()}")
        return out
 