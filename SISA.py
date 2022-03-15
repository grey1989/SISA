import torchvision as tv
import torch.optim as optim
import torch.optim
import torch.nn as nn
import torchvision.transforms as tvtf
import torch
import numpy as np
from sklearn.ensemble import VotingClassifier
# 超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 5
batch_size = 64
learning_rate = 1e-2
# Load a deep CNN pretrained on ImageNet
# Using ResNet18 just to reduce download size, use something deeper
# resnet18 = tv.models.resnet18(pretrained=True)
# resnet18 = resnet18.to(device)
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
#arange = np.arange(0, 5000 )
#ds_train =torch.utils.data.dataset.Subset(ds_train, arange)

#dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False)

# def updateModel():
#     # 最后一层resnet是全连接fc层，in_feature是512,out_feature是1000
#     # 毕竟resnet预训练模型是用imagenet数据集训练的，imagenet有1000个类别
#     cnn_features = resnet18.fc.in_features
#     # 然而cifar是10分类任务，所以需要我们在resnet最后的全连接层上进行微调
#     num_classes = 10
#     resnet18.fc = nn.Sequential(
#         nn.Linear(in_features=cnn_features, out_features=512),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.2),
#         nn.Linear(in_features=512, out_features=10)
#     ).to(device)
#
#     # 可以通过如下方式对不需要训练的网络进行冻结
#     for p in resnet18.parameters():
#         p.requires_grad = False
#     for layer in [resnet18.layer4.parameters(), resnet18.fc.parameters()]:
#         for p in layer:
#             p.requires_grad = True
#     params_non_frozen = filter(lambda p: p.requires_grad, resnet18.parameters())
#     opt = optim.SGD(params_non_frozen, lr=learning_rate, momentum=0.9)
#
#     # 另外一种冻结的方式
#     # opt = torch.optim.SGD([
#     #     dict(params=resnet18.layer1.parameters(), lr=0),
#     #     dict(params=resnet18.layer2.parameters(), lr=0),
#     #     dict(params=resnet18.layer3.parameters(), lr=0),
#     #     dict(params=resnet18.layer4.parameters(), lr=1e-4),
#     #     # layer4这一层微微调一下就行，所以学习率很小
#     #     dict(params=resnet18.fc.parameters()),
#     #     # fc这一层微调，使用0.01学习率
#     # ], lr=learning_rate, momentum=0.9)
#
#     # 如下代码表示学习率调度器。如下设置表示，如果网络更新了5（patience）次，loss值还没有减少，则减少学习率 new_lr = old_lr * factor
#     lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5, )
#     loss_fn = nn.CrossEntropyLoss()

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
# for x in range(epoch):
#     train(resnet18,loss_fn,opt,lr_sched,dl_train,x)
#     test(resnet18,dl_test)


### SISA 参数
model_count = 10
slice_count_per_shared = 5

data_count_per_shard = 50000 // model_count  #10000、5000
data_count_per_slice = data_count_per_shard//slice_count_per_shared  #1000

#### SISA训练

def getTrainDataLoader(shard, slice_num):
    arange = np.arange(shard*data_count_per_shard, shard*data_count_per_shard + (slice_num+1) * data_count_per_slice)
    data_slice =torch.utils.data.dataset.Subset(ds_train, arange)

    return torch.utils.data.DataLoader(data_slice, batch_size=batch_size, shuffle=True)


for shard in range(0, model_count):
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
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=5, )
    loss_fn = nn.CrossEntropyLoss()

#开始训练
    for epoch in range(0, slice_count_per_shared):
        SISA_training_data_loader = getTrainDataLoader(shard, epoch)
        # train for one epoch
        train(resnet18, loss_fn, opt,lr_sched, SISA_training_data_loader,epoch)

        # evaluate on validation set
        #prec1 = validate(test_data_loader, my_model, my_loss_fn)
        test(resnet18,dl_test)
        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        torch.save(resnet18,'./model_10/shard{}-epoch{}.pth'.format((shard+1),(epoch+1)))

# 五个模型进行投票预测
# with torch.no_grad():
#     model1 = torch.load('./model_10/shard{}-epoch{}.pth'.format(1,5))
#     model2 = torch.load('./model_10/shard{}-epoch{}.pth'.format(2, 5))
#     model3 = torch.load('./model_10/shard{}-epoch{}.pth'.format(3, 5))
#     model4 = torch.load('./model_10/shard{}-epoch{}.pth'.format(4, 5))
#     model5 = torch.load('./model_10/shard{}-epoch{}.pth'.format(5, 5))
#     model6 = torch.load('./model_10/shard{}-epoch{}.pth'.format(6, 5))
#     model7 = torch.load('./model_10/shard{}-epoch{}.pth'.format(7, 5))
#     model8 = torch.load('./model_10/shard{}-epoch{}.pth'.format(8, 5))
#     model9 = torch.load('./model_10/shard{}-epoch{}.pth'.format(9, 5))
#     model10 = torch.load('./model_10/shard{}-epoch{}.pth'.format(10, 5))
#     model1.eval()
#     model2.eval()
#     model3.eval()
#     model4.eval()
#     model5.eval()
#     model6.eval()
#     model7.eval()
#     model8.eval()
#     model9.eval()
#     model10.eval()
#     total_cnt = 0
#     for i,(data,target) in enumerate(dl_test):
#         logit1 = model1(data.to(device))
#         logit2 = model2(data.to(device))
#         logit3 = model3(data.to(device))
#         logit4 = model4(data.to(device))
#         logit5 = model5(data.to(device))
#         logit6 = model6(data.to(device))
#         logit7 = model7(data.to(device))
#         logit8 = model8(data.to(device))
#         logit9 = model9(data.to(device))
#         logit10 = model10(data.to(device))
#         predict = logit1.max(1)[1]
#         logit = logit1+logit2+logit3+logit4+logit5+logit6+logit7+logit8+logit9+logit10
#         logit = torch.div(logit,model_count)
#         predict = logit.max(1)[1]
#
#         # logit = (data.to(device))
#         # predict = logit.max(1)[1]
#         # if i == 0:
#         #   print (predict)
#         #   print (target)
#         #   break
#         cnt = torch.sum(predict==target.to(device))
#         total_cnt = total_cnt + cnt
#     print ('correct rate is {}'.format(total_cnt/10000))