import torchvision.transforms as transforms
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root='.\CLFAR_10',
    train=True,
    transform=transform,
    download=True
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=4,
    shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root='.\CLFAR_10',
    train=False,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=4,
    shuffle=False,
)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data,label = trainset[100]
print(data.size()[0])
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5),#6*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#6*14*14
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),#16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#16*5*5
        )
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net =Net()
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    runningloss = 0
    i=0
    for data,label in trainloader:
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        runningloss = runningloss + loss.item()
        i=i+1
        if i % 2000 == 1999: # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, runningloss / 2000))
            runningloss = 0.0
            i=0

correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数


# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print(' %d ' % (100 * correct / total))