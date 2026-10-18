import torch
import torch.nn as nn           #包含各种层和损失函数
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms      #导入图像变换工具，用于调整大小，归一化操作
import matplotlib.pyplot as plt                  #画图
from pyexpat import model

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005
NUM_CLASSES = 43
IMAGE_SIZE = 32

#检测GPU可用性，有NVIDIA且安装了CUDA，使用CUBA，否则CPU
#保证速度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")

#预处理
"""transform = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE),
     transforms.ToTensor(),                  #转换为Tensor格式，归一化
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]    #标准化
    )
"""
transform = transforms.Compose([             #按顺序经过每一步
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),       #以 50% 的概率随机水平翻转图片（数据增强）
    transforms.RandomRotation(10),           #随机旋转图片，旋转角度在 -10° 到 +10° 之间（数据增强）
    transforms.RandomAffine(0,translate=(0.1,0.1)),#随机平移图片，水平和垂直方向最多平移 10%（数据增强）
    transforms.ToTensor(),                   #转成pytorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载 GTSRB 训练集
trainset = torchvision.datasets.GTSRB(root="./data",split="train",download=True,transform=transform)
#创建训练集数据加载器
trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.GTSRB(root="./data",split="test",download=True,transform=transform)
testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

#构建CNN模型
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()        #初始化父类(自己把基础设施建好）
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),                  #批归一化（加速训练，防止梯度消失）
            nn.ReLU(inplace=True),               #inplace省内存
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*2*2,512),         #映射到512维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                           #随机丢弃30%神经元，防止过拟合

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, NUM_CLASSES),     #输出层：映射到43个类别
        )

    def forward(self, x):
        x = self.features(x)                       #通过卷积层提取特征
        x = x.view(-1, 256*2*2)                # 展平操作：将多维 Tensor 变为一维向量，准备输入全连接层
        x = self.classifier(x)
        return x                                   #返回原始 logits（未归一化的概率值）

model = ImprovedCNN().to(device)

#损失函数与优化器
criterion = nn.CrossEntropyLoss()  #包含softmax
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#训练与评估循环
train_losses, train_accs = [], []
val_losses, val_accs = [], []      #创建 4 个空列表，用来记录每个 epoch 的训练和验证结果

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct,total = 0,0            #每个 epoch 开始时，把统计变量清零，为新一轮的累加做准备。

    for inputs,labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)  #数据移入GPU

        optimizer.zero_grad()      #清空上一轮残留的梯度
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #前向传播：计算预测值 计算损失 反向传播 更新参数
        running_loss += loss.item()*inputs.size(0)  #统计计算准确率
        predicted = torch.argmax(outputs,dim=1)      #probability
        total +=labels.size(0)                      #计算当前这个 batch 的总样本数和预测正确的个数，并把它们累加到整个 epoch 的总分里。
        correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss/len(trainloader)
    avg_train_acc = 100*correct / total

    #验证阶段
    model.eval()           #设置模型为评估模式（关闭 Dropout，固定 BatchNorm）
    val_loss,val_correct,val_total=0.0,0,0

    with torch.no_grad():  # 评估时不需要计算梯度，节省显存和计算资源
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(testloader.dataset)
    avg_val_acc = 100 * val_correct / val_total        #*100是为了转换为百分比显示

    #记录数据
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)
    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)

    #打印当前进度
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")

    # 绘制 Loss 曲线
plt.figure(figsize=(12,5))   #创建一个新画布，并设置画布大小为宽 12 英寸、高 5 英寸。
plt.subplot(1, 2, 1)  # 创建 1行2列 的第1个子图
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

    # 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)  # 创建 1行2列 的第2个子图
plt.plot(train_accs, label='Train Accuracy', color='blue')
plt.plot(val_accs, label='Val Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图表

print(f"\n最终验证准确率：{val_accs[-1]:.2f}%")