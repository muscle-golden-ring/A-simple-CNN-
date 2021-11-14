import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     #
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/jhub/students/data/course8/', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=4)
#加载测试集
testset = torchvision.datasets.CIFAR10(root='/jhub/students/data/course8/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=4)

###


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)				# 卷积层：3通道到6通道，卷积5*5
        self.conv2 = nn.Conv2d(6, 16, 5)			# 卷积层：6通道到16通道，卷积5*5

        self.pool = nn.MaxPool2d(2, 2)				# 池化层，在2*2窗口上进行下采样

		# 三个全连接层 ：16*5*5 -> 120 -> 84 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

	# 定义数据流向
    def forward(self, x):
        x = F.relu(self.conv1(x))        # F.relu 是一个常用的激活函数
        x = self.pool(x)
        x = F.relu(self.conv2(x))    
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)			# 变换数据维度为 1*(16*5*5)，-1表示根据后面推测

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
def train(model, criterion, optimizer, epochs):
    since = time.time()
    net = Net()
    best_acc = 0.0      # 记录模型测试时的最高准确率
    best_model_wts = copy.deepcopy(model.state_dict())  # 记录模型测试出的最佳参数

    for epoch in range(epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch+1, epochs))

        # 训练模型
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播，计算损失
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每1000批图片打印训练数据
            if (i != 0) and (i % 1000 == 0):
                print('step: {:d},  loss: {:.3f}'.format(i, running_loss/1000))
                running_loss = 0.0

        # 每个epoch以测试数据的整体准确率为标准测试一下模型
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        if acc > best_acc:      # 当前准确率更高时更新
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('最高准确率: {}%'.format(100 * best_acc))

    # 返回测试出的最佳模型
    model.load_state_dict(best_model_wts)
    return model
def imshow(img):
    # 输入数据：torch.tensor[c, h, w]
    img = img * 0.5 + 0.5     # 反归一
    npimg = np.transpose(img.numpy(), (1, 2, 0))    # [c, h, w] -> [h, w, c]
    plt.imshow(npimg)
    plt.show()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def main():

 net = Net()
 net.to(DEVICE)

# 使用分类交叉熵 Cross-Entropy 作损失函数，动量SGD做优化器
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练10个epoch
 net = train(net, criterion, optimizer, 10)
# 保存模型参数
 torch.save(net.state_dict(), 'net_dict.pt')
 


 net = Net()
 net.load_state_dict(torch.load('net_dict.pt'))  # 加载各层参数
 net.to(DEVICE)

# 整体正确率
 correct = 0
 total = 0
 with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

 print('整体准确率: {}%'.format(100 * correct / total))

 print('=' * 30)

# 每一个类别的正确率
 class_correct = list(0. for i in range(10))
 class_total = list(0. for i in range(10))
 with torch.no_grad():
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

 for i in range(10):
    print('{}的准确率 : {:.2f}%'.format(classes[i], 100 * class_correct[i] / class_total[i]))
 testdata = iter(testloader)
 images, labels = testdata.next()
 imshow(torchvision.utils.make_grid(images))
 print('真实类别: ', ' '.join('{}'.format(classes[labels[j]]) for j in range(labels.size(0))))

# 预测是10个标签的权重，一个类别的权重越大，神经网络越认为它是这个类别，所以输出最高权重的标签。
 outputs = net(images)
 _, predicted = torch.max(outputs, 1)
 print('预测结果: ', ' '.join('{}'.format(classes[predicted[j]]) for j in range(labels.size(0))))
if __name__ == '__main__':
	main()
