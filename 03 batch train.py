import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)  1~10
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)  10~1

torch_dataset = Data.TensorDataset(x, y)
#DataLoader 是 torch 给你用来包装你的数据的工具
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training  洗牌，是否在训练时随机打乱再开始训练
    num_workers=2,              # subprocesses for loading data  多线程
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times   把数据集完整训练完一次，叫一个epoch
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()