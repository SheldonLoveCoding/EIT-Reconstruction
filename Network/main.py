import torch
from myNN import EITNet
from utils import dataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter



def prepareDataset(ratio = 0.8):
    path_1 = '../Dataset/data/allRegionDataset_v1_train_1.mat'
    path_2 = '../Dataset/data/allRegionDataset_v2_train_1.mat'
    dataset_1 = dataLoader(path_1)
    dataset_2 = dataLoader(path_2)
    dataset_all = np.vstack([dataset_1, dataset_2])
    np.random.shuffle(dataset_all) # 打乱顺序
    train_num = int(np.floor(dataset_all.shape[0] * ratio))

    train_dataset = dataset_all[0: train_num, :]
    test_dataset = dataset_all[train_num: , :]
    return train_dataset, test_dataset


def train(epoch):
    model.train()
    train_loss = 0

    loop_num = train_dataset.shape[0] // batch_size
    # print(loop_num)
    for i in range(loop_num):
        # 随机抽样
        idx = np.random.randint(0, train_dataset.shape[0], batch_size)
        td_volt = train_dataset[idx, 0 : 208].view(-1, 208)
        image_label = train_dataset[idx, 208 : ].view(batch_size, -1, image_size, image_size)
        # 将掩码沿着batch的维度升维
        mask_label_batch = mask_label.repeat(batch_size, 1, 1, 1)

        optimizer.zero_grad()

        # 得到网络的输出
        image_pre = model(td_volt, image_size)
        # 利用掩码将无关的因素去掉
        image_pre_marked = torch.mul(image_pre, mask_label_batch)
        # 计算loss
        loss = loss_function(image_label, image_pre_marked)
        # print(loss)
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()

        train_loss += loss * batch_size

    train_loss = train_loss / train_dataset.shape[0]
    writer.add_scalar('Loss/train', train_loss, epoch)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def test(epoch):
    model.eval()

    test_loss = 0
    loop_num = test_dataset.shape[0] // batch_size
    # print(loop_num)
    for i in range(loop_num):
        # 随机抽样
        idx = np.random.randint(0, test_dataset.shape[0], batch_size)
        td_volt = test_dataset[idx, 0: 208].view(-1, 208)
        image_label = test_dataset[idx, 208:].view(batch_size, -1, image_size, image_size)
        # 将掩码沿着batch的维度升维
        mask_label_batch = mask_label.repeat(batch_size, 1, 1, 1)

        # 得到网络的输出
        image_pre = model(td_volt, image_size)
        # 利用掩码将无关的因素去掉
        image_pre_marked = torch.mul(image_pre, mask_label_batch)
        # 计算loss
        loss = loss_function(image_label, image_pre_marked)

        test_loss += loss * batch_size

    test_loss = test_loss / train_dataset.shape[0]
    writer.add_scalar('Loss/test', test_loss, epoch)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, test_loss))

if __name__ == "__main__":
    voltage_num = 208
    image_size = 64
    batch_size = 32
    learning_rate = 1e-4
    epochs = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter('./EITNN_train_test_log/log2')

   # 加载数据集，训练：测试 = 4：1 image 在最终可视化的时候需要转置，中间运算不需要
    [train_dataset_nan, test_dataset_nan] = prepareDataset()
    train_dataset_nan = torch.Tensor(train_dataset_nan).to(device)
    test_dataset_nan = torch.Tensor(test_dataset_nan).to(device)
    image_temp = train_dataset_nan[0, 208 : ]
    image_temp = image_temp.view(image_size, image_size)
    # 计算掩码
    mask_label = (~torch.isnan(image_temp)).int()

    # 将为nan的值都改成0
    train_dataset = torch.where(torch.isnan(train_dataset_nan), torch.full_like(train_dataset_nan, 0), train_dataset_nan)
    test_dataset = torch.where(torch.isnan(test_dataset_nan), torch.full_like(test_dataset_nan, 0), test_dataset_nan)


    print(train_dataset.shape)
    print(test_dataset.shape)
    model = EITNet(voltage_num, image_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        train(epoch)
        test(epoch)

    # 确保tensorboard的记录的内容写入磁盘
    writer.flush()

    print("Training is done")
    # 先建立路径
    savepath = './model_saved/eit_nn.pth'
    # 保存:可以是pth文件(参数)
    torch.save(model.state_dict(), savepath)
    print("Model is saved! ")
















