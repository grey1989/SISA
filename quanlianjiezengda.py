import torchvision as tv
import torch.optim as optim
import torch.optim
import torch.nn as nn
import torchvision.transforms as tvtf
import torch
import numpy as np

# 超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 20
batch_size = 64
learning_rate = 1e-2
# Load a deep CNN pretrained on ImageNet
# Using ResNet18 just to reduce download size, use something deeper
resnet18 = tv.models.resnet18(pretrained=True)
resnet18 = resnet18.to(device)
# resnet18

tf = tvtf.Compose([
    # 将32*32*3大小图片，resize成224*224*3大小
    tvtf.Resize(224),
    tvtf.ToTensor(),
    # 以下对3通道进行不同的正态化，也就是遵循imagenet数据集的处理方式。
    # 因为本身resnet的训练就是拿imagenet数据集进行训练的，那imagenet做了什么预处理，我们也得跟上不是
    tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载Cifar数据集，训练集5万图片，测试集1万图片。10分类任务
ds_train = tv.datasets.CIFAR10(root='./data', download=True, train=True, transform=tf)
ds_test = tv.datasets.CIFAR10(root='./data', download=True, train=False, transform=tf)
arange = np.arange(0, 5000 )
ds_train =torch.utils.data.dataset.Subset(ds_train, arange)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False)

# 最后一层resnet是全连接fc层，in_feature是512,out_feature是1000
# 毕竟resnet预训练模型是用imagenet数据集训练的，imagenet有1000个类别
cnn_features = resnet18.fc.in_features
# 然而cifar是10分类任务，所以需要我们在resnet最后的全连接层上进行微调
num_classes = 10
resnet18.fc =  nn.Sequential(
    nn.Linear(in_features=cnn_features, out_features=1024),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1024, out_features=1024),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
nn.Linear(in_features=1024, out_features=10)
).to(device)

# 可以通过如下方式对不需要训练的网络进行冻结
for p in resnet18.parameters():
    p.requires_grad = False
for layer in [resnet18.layer4.parameters(),resnet18.fc.parameters()]:
  for p in layer:
    p.requires_grad = True
params_non_frozen = filter(lambda p: p.requires_grad, resnet18.parameters())
opt = optim.SGD(params_non_frozen, lr=learning_rate, momentum=0.9)


# 另外一种冻结的方式
# opt = torch.optim.SGD([
#     dict(params=resnet18.layer1.parameters(), lr=0),
#     dict(params=resnet18.layer2.parameters(), lr=0),
#     dict(params=resnet18.layer3.parameters(), lr=0),
#     dict(params=resnet18.layer4.parameters(), lr=1e-4),
#     # layer4这一层微微调一下就行，所以学习率很小
#     dict(params=resnet18.fc.parameters()),
#     # fc这一层微调，使用0.01学习率
# ], lr=learning_rate, momentum=0.9)

# 如下代码表示学习率调度器。如下设置表示，如果网络更新了5（patience）次，loss值还没有减少，则减少学习率 new_lr = old_lr * factor
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5,)
loss_fn = nn.CrossEntropyLoss()

def train(model, loss_fn , opt, lr_sched, dl_train,x):
    # Same as regular classifier traning, just call lr_sched.step() every epoch.
    # for x in range(epoch):
      running_corrects = 0
      for i,(data,target) in enumerate(dl_train):
        data = data.to(device)
        target = target.to(device)
        logit = model(data)
        loss = loss_fn(logit,target)
        resnet18.zero_grad()
        loss.backward()
        opt.step()
        lr_sched.step(loss)
        _, preds = torch.max(logit, 1)
        running_corrects += torch.sum(preds == target.data)
        if i%100 == 0:
          print ('epoch {} : running about {} images and loss equals {}'.format(x,i*batch_size,loss))
      print('train_correct_rate {}'.format(running_corrects/5000))



def test(model,dl_test):
  with torch.no_grad():
      model.eval()
      total_cnt = 0
      for i,(data,target) in enumerate(dl_test):
        logit = model(data.to(device))
        predict = logit.max(1)[1]
        # if i == 0:
        #   print (predict)
        #   print (target)
        #   break
        cnt = torch.sum(predict==target.to(device))
        total_cnt = total_cnt + cnt
      print ('correct rate is {}'.format(total_cnt/10000))
for x in range(epoch):
    train(resnet18,loss_fn,opt,lr_sched,dl_train,x)
    test(resnet18,dl_test)