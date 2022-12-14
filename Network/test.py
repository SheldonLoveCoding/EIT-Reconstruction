import torch
from myNN import EITNet
from utils import dataLoader
import numpy as np
import matplotlib.pyplot as plt

voltage_num = 208
image_size = 64
savepath = './model_saved/eit_nn.pth'

datapath = '../Dataset/data/test_dataset'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EITNet(voltage_num, image_size).to(device)
model.load_state_dict(torch.load(savepath))
model.eval()


test_dataset_nan = torch.load('../Dataset/data/test_dataset_nan')
test_dataset_nan = torch.Tensor(test_dataset_nan)
image_temp = test_dataset_nan[0, 208 : ]
image_temp = image_temp.view(image_size, image_size)
# 计算掩码
mask_label = (~torch.isnan(image_temp)).int().to(device)
# 将掩码沿着batch的维度升维
nan_matrix = torch.Tensor(np.full((image_size, image_size), np.nan)).to(device)
mask_label_nan = torch.where(mask_label == 0, nan_matrix, mask_label.float())

test_num = 1
td_volt_test = test_dataset_nan[2, 0:208].view(-1, 208).to(device)
image_temp_test = test_dataset_nan[2, 208:].view(image_size, image_size).T

# 模型预测的图像
with torch.no_grad():
    image_pre_test = model(td_volt_test, image_size).view(image_size, image_size)

# 利用掩码将无关的因素去掉
image_pre_test_marked = torch.mul(image_pre_test, mask_label_nan).T
plt.subplot(1, 2, 1)
plt.imshow(image_temp_test)

plt.subplot(1, 2, 2)
plt.imshow(image_pre_test_marked.detach().cpu().view(image_size, image_size))

plt.show()