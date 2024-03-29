import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)  将一维数据变成二维数据
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
plt.scatter(x.data.numpy(), y.data.numpy()) #散点图
plt.show()


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer  隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer  输出层线性输出

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # activation function for hidden layer 激励函数(隐藏层的线性值)
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network  输入只有一个下，输出为y
print(net)  # net architecture   打印出net的结构

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  #优化器 随机梯度下降 传入net的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss 预测值和真实值的误差计算公式 (均方差)

plt.ion()   # something about plotting 开启交互模式

for t in range(200):#训练步数200步
    prediction = net(x)     # input x and predict based on x  喂给net训练数据x,输出预测值

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)   计算预测值和真实值y 两者的误差

    optimizer.zero_grad()   # clear gradients for next train  清空上一步的残余更新参数值
    loss.backward()         # backpropagation, compute gradients  误差反向传播, 计算参数更新值
    optimizer.step()        # apply gradients   将参数更新值施加到net的parameters 上

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy()) #散点图
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5) #画线
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


