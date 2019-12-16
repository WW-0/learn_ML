import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
#每个元素是从 均值=2*n_data中对应位置的取值，标准差为1的正态分布中随机生成的
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#输入数据1，输入数据2，c色彩颜色顺序，s绘制的点面积(默认36面积，标量就是相同大小，向量则和输入数据意义对应)，lw线宽，cmap色彩盘
plt.show()


class Net(torch.nn.Module): #参见有莫烦快速搭建法
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)  #输出值，但是不是预测值，预测值需要另外计算
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network 几个类别就几个out
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted   输出的是概率

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    #print(out)
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1] #返回Tensor中所有元素的最大值，按维度dim 返回最大值
        #torch.max(a,1) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一行的列索引）
        #torch.max(a,0) 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
        #torch.max()[0]，只返回最大值的值
        #troch.max()[1]，只返回最大值的索引
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)#计算正确率 astype numpy数据类型转换
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.3)
#模型保存方式
torch.save(net,'net.pkl')#保存整个网络
torch.save(net.state_dict(),'net_parameters.pkl') #只保存网络中的参数，但是图节点不保留，占内存少

#提取网络,这种方式将会提取整个神经网络, 网络大的时候可能会比较慢.
def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)
#提取网络,这种方式将会提取所有的参数, 然后再放到你的新建网络中.
def restore_params():
    # 新建 net3  快速搭建法
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    # 将保存的参数复制到 net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

plt.ioff()
plt.show()