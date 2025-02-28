import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot model training result')
parser.add_argument('--dir', default='trained-model-AHDR')
args = parser.parse_args()

## curve_loss.png ##

# CSV 파일 읽기 (첫 번째 행이 헤더라면 skiprows=1)
file_path = args.dir + '/plot_data.txt' 
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

plt.savefig(f"{args.dir}/curve_loss.png", dpi=300, bbox_inches="tight")  # 고해상도 저장


## curve_psnr.png ##

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

plt.savefig(f"{args.dir}/curve_psnr.png", dpi=300, bbox_inches="tight")  # 고해상도 저장


## curve_loss_batch.png ##
file_path = args.dir + '/loss.txt' 
data = np.loadtxt(file_path, delimiter=",", skiprows=0)

# CSV에서 데이터 읽기 
loss_batch = data[:, 2]
iter = len(loss_batch)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(loss_batch, label="train_loss", color="red",  linewidth=2)

# 그래프 꾸미기
plt.xlabel("Iter")
plt.ylabel("Train loss")
plt.title("Train loss Curve")
plt.legend()
plt.grid(True)

plt.savefig(f"{args.dir}/curve_train_loss.png", dpi=300, bbox_inches="tight")  # 고해상도 저장