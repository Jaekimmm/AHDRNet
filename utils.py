import os
import torch
from torch.nn import init
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def model_load(model, trained_model_dir, model_file_name):
    model_path = os.path.join(trained_model_dir, model_file_name)
    # trained_model_dir + model_file_name    # '/modelParas.pkl'
    model.load_state_dict(torch.load(model_path))
    return model

class EarlyStopping:    # copy from freeSoul
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.best_update = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_update = True
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                self.best_update = False
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score.item() - score.item()):.5f}')
                    
                    
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score.item() - score.item()):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
            
# Evaluation Metrics (copy from freeSoul)
def psnr(im0, im1):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0

        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.

        """
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

def normalized_psnr(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.

        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.

        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.

        """
    return psnr(im0/norm, im1/norm)

def mu_tonemap(hdr_image, mu=5000):
    """ This function computes the mu-law tonemapped image of a given input linear HDR image.

    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        mu (float): Parameter controlling the compression performed during tone mapping.

    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.

    """
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value and then computes
    the mu-law tonemapped image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
        mu (float): Parameter controlling the compression performed during tone mapping.

    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.

    """
    return mu_tonemap(hdr_image/norm_value, mu)

def psnr_norm_mu(im0,im1,norm):
    return psnr(norm_mu_tonemap(im0,norm),norm_mu_tonemap(im1,norm))

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value, afterwards bounds the
    HDR image values by applying a tanh function and afterwards computes the mu-law tonemapped image.

        the mu-law tonemapped image.
        Args:
            hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
            norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
            mu (float): Parameter controlling the compression performed during tone mapping.

        Returns:
            np.ndarray (): Returns the mu-law tonemapped image.

        """
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    """ This function computes Peak Signal to Noise Ratio (PSNR) between the mu-law computed images from two non-linear
    HDR images.

            Args:
                hdr_nonlinear_ref (np.ndarray): HDR Reference Image after gamma correction, used for the percentile norm
                hdr_nonlinear_res (np.ndarray: HDR Estimated Image after gamma correction
                percentile (float): Percentile to to use for normalization
                gamma (float): Value used to linearized the non-linear images

            Returns:
                np.ndarray (): Returns the mean mu-law PSNR value for the complete image.

            """
    hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))

def save_plot(trained_model_dir):
    # CSV 파일 읽기 (첫 번째 행이 헤더라면 skiprows=1)
    file_path = trained_model_dir + '/plot_data.txt' 
    data = np.loadtxt(file_path, delimiter=",", skiprows=0)

    # CSV에서 데이터 읽기 (0번째 열: epoch, 1번째 열: train_loss, 2번째 열: val_loss)
            # fplot.write(f'{epoch},{train_loss},{valid_loss},{psnr},{psnr_mu}\n')
    epochs = data[1:, 0]
    train_loss = data[1:, 1]
    val_loss = data[1:, 2]

    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="red", linestyle="dashed", linewidth=2)

    # 그래프 꾸미기
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{trained_model_dir}/curve_loss.png", dpi=300, bbox_inches="tight")  # 고해상도 저장


    # CSV에서 데이터 읽기 (0번째 열: epoch, 1번째 열: train_loss, 2번째 열: val_loss)
            # fplot.write(f'{epoch},{train_loss},{valid_loss},{psnr},{psnr_mu}\n')
    epochs = data[:, 0]
    psnr_l = data[:, 3]
    psnr_m = data[:, 4]

    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnr_l, label="PSNR", color="blue", linewidth=2)
    plt.plot(epochs, psnr_m, label="PSNR_mu", color="red",  linewidth=2)
    plt.axhline(y=43.7708, color="blue", linestyle="--", linewidth=2, label="PSNR (Baseline)")
    plt.axhline(y=47.1223, color="red",  linestyle="--", linewidth=2, label="PSNR_mu (Baseline)")

    # 그래프 꾸미기
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{trained_model_dir}/curve_psnr.png", dpi=300, bbox_inches="tight")  # 고해상도 저장
    