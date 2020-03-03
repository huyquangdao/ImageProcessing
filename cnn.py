import torch
import torchvision
import numpy as np
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader


class CNNModel(torch.nn.Module):

    def __init__(self,n_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.max_pool2d_1 = torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.max_pool2d_2 = torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.linear_1 = torch.nn.Linear(in_features=7*7*128,out_features=128)
        self.drop_out = torch.nn.Dropout(p=0.5)
        self.output = torch.nn.Linear(in_features=128,out_features=n_classes)

    def forward(self,input):

        out = self.conv1(input)
        out = torch.relu(out)
        out = self.max_pool2d_1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.max_pool2d_2(out)
        out = out.view(out.size(0),-1).contiguous()
        out = self.linear_1(out)
        out = self.drop_out(out)
        out = self.output(out)

        return out


def cal_accuracy(logits,targets):
    pred = torch.argmax(logits,dim=-1)
    correct = torch.sum(pred==targets)
    acc = correct.double()/logits.size(0)
    return acc


def cal_mean(lis):
    return sum(lis)/len(lis)


def train(epoch,batch_size,lr):

    model = CNNModel(n_classes=10)
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/mnist', train=True, download=False,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size, shuffle=True,num_workers=2)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/mnist', train=False, download=False,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    loss_func = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0")

    model.to(device)

    for i in range(epoch):

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        model.train()
        for batch_idx,(batch_data,batch_target) in enumerate(train_loader):

            print(batch_data.shape)
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = loss_func(input=output,target=batch_target)

            train_loss.append(loss.item())
            acc = cal_accuracy(output,batch_target)
            train_acc.append(acc)

            loss.backward()
            optimizer.step()

        model.eval()
        for batch_idx,(batch_data,batch_target) in enumerate(test_loader):

            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            output = model(batch_data)
            loss = loss_func(input=output,target=batch_target)

            val_loss.append(loss.item())
            acc = cal_accuracy(output,batch_target)
            val_acc.append(acc)

        torch.save(model.state_dict(), 'pretrained/model.pth')

        print("epoch:{0}, train_loss:{1}, train_acc:{2}, val_loss:{3}, val_acc:{4}".format(i,cal_mean(train_loss),cal_mean(train_acc),
                                                                                           cal_mean(val_loss),cal_mean(val_acc)))


def predict_on_images(model,images):
    images = images/255
    images_tensor = torch.from_numpy(images).type(torch.float32)
    images_tensor = torch.unsqueeze(images_tensor,dim=1)
    output = model(images_tensor)
    predict_classes = torch.argmax(output,dim=-1).numpy()
    print(predict_classes)
    return predict_classes

if __name__ == '__main__':
    train(5,32,0.001)


