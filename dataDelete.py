import torchvision as tv
import torch.optim as optim
import torch.optim
import torch.nn as nn
import torchvision.transforms as tvtf
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_count = 5
slice_count_per_shared = 10
batch_size = 64
learning_rate = 1e-2
data_count_per_shard = 50000 // model_count  #10000、5000
data_count_per_slice = data_count_per_shard//slice_count_per_shared  #1000
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

def deleteDataByNum(num):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []

    for x in num:
        if x>=0&x<10000:
            list1.append(x)
        elif x>=10000&x<20000:
            list2.append(x)
        elif x>=20000&x<30000:
            list3.append(x)
        elif x>=30000&x<40000:
            list4.append(x)
        else:
            list5.append(x)
    if list1:
        list1.sort()
        deleteMethod(list1)
    if list2:
        list2.sort()
        deleteMethod(list2)
    if list3:
        list3.sort()
        deleteMethod(list3)
    if list4:
        list4.sort()
        deleteMethod(list4)
    if list5:
        list5.sort()
        deleteMethod(list5)


def deleteMethod(list):
    shard = list[0] // data_count_per_shard   #32311、3
    minNum = list[0] - shard*data_count_per_shard #2600\2311
    minNum = minNum // data_count_per_slice #2000
    resnet18, lr_sched, loss_fn, opt = getModel(shard,minNum)
    for epoch in range(minNum,data_count_per_shard):
        training_data_loader,flagList = getTrainDataLoader(shard,epoch,list)




def getTrainDataLoader(shard, slice_num, list):
    arange = np.arange(shard*data_count_per_shard, shard*data_count_per_shard + (slice_num+1)
                       * data_count_per_slice)
    for i in list:
        if(list[i]>=shard*data_count_per_shard & list[i]<shard*data_count_per_shard + (slice_num+1)
                       * data_count_per_slice):
            arange = np.delete(arange,list[i])
    data_slice =torch.utils.data.dataset.Subset(ds_train, arange)

    return torch.utils.data.DataLoader(data_slice, batch_size=batch_size, shuffle=True)

def getModel(model_num,minNum):
    if minNum*1000 <1000:
        resnet18 = tv.models.resnet18(pretrained=True)
        resnet18 = resnet18.to(device)
        # 最后一层resnet是全连接fc层，in_feature是512,out_feature是1000
        # 毕竟resnet预训练模型是用imagenet数据集训练的，imagenet有1000个类别
        cnn_features = resnet18.fc.in_features
        # 然而cifar是10分类任务，所以需要我们在resnet最后的全连接层上进行微调
        num_classes = 10
        resnet18.fc = nn.Sequential(
            nn.Linear(in_features=cnn_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=10)
        ).to(device)

        # 可以通过如下方式对不需要训练的网络进行冻结
        for p in resnet18.parameters():
            p.requires_grad = False
        for layer in [resnet18.layer4.parameters(), resnet18.fc.parameters()]:
            for p in layer:
                p.requires_grad = True
        params_non_frozen = filter(lambda p: p.requires_grad, resnet18.parameters())
        opt = optim.SGD(params_non_frozen, lr=learning_rate, momentum=0.9)

        # 如下代码表示学习率调度器。如下设置表示，如果网络更新了5（patience）次，loss值还没有减少，则减少学习率 new_lr = old_lr * factor
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5, )
        loss_fn = nn.CrossEntropyLoss()
    else:
        resnet18 = torch.load('./model/shard{}-epoch{}.pth'.format(model_num, minNum))
        resnet18.to(device)
        # 可以通过如下方式对不需要训练的网络进行冻结
        for p in resnet18.parameters():
            p.requires_grad = False
        for layer in [resnet18.layer4.parameters(), resnet18.fc.parameters()]:
            for p in layer:
                p.requires_grad = True
        params_non_frozen = filter(lambda p: p.requires_grad, resnet18.parameters())
        opt = optim.SGD(params_non_frozen, lr=learning_rate, momentum=0.9)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5, )
        loss_fn = nn.CrossEntropyLoss()
    return resnet18, lr_sched, loss_fn,opt