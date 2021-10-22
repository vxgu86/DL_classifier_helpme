# -*- coding:utf-8 -*-
'''

datasets
dataloader
transforms

'''
import torch
import torchvision
import torchvision.transforms as transforms
import os.path

import sys
sys.path.append("../")

from common import generate_unique_logpath,ModelCheckpoint,compute_mean_std


dataset_dir=os.path.join("../pytroch_exp/data",'FashionMNIST')

print(dataset_dir)  
valid_ratio=0.2  #80/20 split  train/valid

train_valid_dataset=torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                      train=True,
                                                      transform=None, #transoforms.ToTensor(),
                                                      download=True)

nb_train=int((1.0-valid_ratio)*len(train_valid_dataset))
nb_valid=int(valid_ratio      *len(train_valid_dataset))
train_dataset,valid_dataset=torch.utils.data.dataset.random_split(train_valid_dataset,[nb_train,nb_valid])


test_dataset=torchvision.datasets.FashionMNIST(root=dataset_dir,
                                               transform=None,#transforms.ToTensor(),
                                               train=False)

class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self,base_dataset,transform):
        self.base_dataset=base_dataset
        self.transform=transform

    def __getitem__(self, index):
        img,target=self.base_dataset[index]
        return self.transform(img),target

    def __len__(self):
        return len(self.base_dataset)

train_augment_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(degrees=10,translate=(0.1,0.1)),
    transforms.ToTensor()
])

num_threads=0  #use 4 cpu threads
batch_size=128 #minibatch of 128
normalize=True
if normalize:
    normalize_dataset=DatasetTransformer(train_dataset,transforms.ToTensor())
    normalize_loader=torch.utils.data.DataLoader(dataset=normalize_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=num_threads)

    mean_train_tensor,std_train_tensor=compute_mean_std(normalize_loader)
    train_augment_transform=transforms.Compose([
        train_augment_transform,
        transforms.Lambda(lambda x:(x-mean_train_tensor)/std_train_tensor)
    ])
    print("----normalize----")

train_dataset=DatasetTransformer(train_dataset,train_augment_transform)
valid_dataset=DatasetTransformer(valid_dataset,transforms.ToTensor())
test_dataset =DatasetTransformer(test_dataset ,transforms.ToTensor())



############################################################3dataloader

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,  
                                         num_workers=num_threads)
valid_loader=torch.utils.data.DataLoader(dataset=valid_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_threads)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_threads)
print("train set contains {} iamges, in {} batches".format(len(train_loader.dataset),len(train_loader)))
print("validation set contains {} images, in {} batches".format(len(valid_loader.dataset),len(valid_loader)))
print("test set contains {} images, in {} batches".format(len(test_loader.dataset),len(test_loader)))


#############build
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self,input_size,num_classes):
        super(LinearNet,self).__init__()
        self.input_size=input_size
        self.classifier=nn.Linear(self.input_size,num_classes)

    def forward(self,x):
        x=x.view(x.size()[0],-1)
        y=self.classifier(x)
        return y

model=LinearNet(1*28*28,10)

def linear_relu(dim_in,dim_out):
    return [nn.Linear(dim_in,dim_out),
            nn.ReLU(inplace=True)]

class FullyConnected(nn.Module):
    def __init__(self,input_size,num_classes):
        super(FullyConnected,self).__init__()
        self.classifier=nn.Sequential(
            *linear_relu(input_size,256),
            *linear_relu(256,256),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x=x.view(x.size()[0],-1)
        y=self.classifier(x)
        return y

class FullyConnectedRegularized(nn.Module):
    def __init__(self,input_size,num_classes,l2_reg):
        super(FullyConnectedRegularized,self).__init__()
        self.l2_reg=l2_reg
        self.lin1=nn.Linear(input_size,256)
        self.lin2=nn.Linear(256,256)
        self.lin3=nn.Linear(256,num_classes)

    def penalty(self):
        return self.l2_reg*(self.lin1.weight.norm(2)+self.lin2.weight.norm(2)+self.lin3.weight.norm(2))

    def forward(self,x):
        x=x.view(x.size()[0],-1)
        x=nn.functional.relu(self.lin1(x))
        x=nn.functional.relu(self.lin2(x))
        y=self.lin3(x)
        return y

#model=FullyConnectedRegularized(1*28*28,10,1e-3)

def conv_relu_maxp(in_channels,out_channels,ks):
    return [nn.Conv2d(in_channels,out_channels,kernel_size=ks,
                      stride=1,padding=int((ks-1)/2),bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]
def dropout_linear_relu(dim_in,dim_out,p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in,dim_out),
            nn.ReLU(inplace=True)]

class VanillaCNN(nn.Module):
    def __init__(self,num_classes):
        super(VanillaCNN, self).__init__()
        self.features=nn.Sequential(
            *conv_relu_maxp(1,16,5),
            *conv_relu_maxp(16,32,5),
            *conv_relu_maxp(32,64,5)
        )


        prob_tensor=torch.zeros((1,1,28,28))
        out_features=self.features(prob_tensor)
        print("out_features1:",out_features.shape)
        out_features=self.features(prob_tensor).view(-1)
        print("out_features2:",out_features.shape)

        self.classifier=nn.Sequential(
            *dropout_linear_relu(out_features.shape[0],128,0.5),
            *dropout_linear_relu(128,256,0.5),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size()[0],-1)
        #print("x in forward before classifier:",x.shape)
        #x torch.Size([128, 576])
        y=self.classifier(x)
        return y
model=VanillaCNN(10)

use_gpu=torch.cuda.is_available()
if use_gpu:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

model.to(device)

#unit test

f_loss=torch.nn.CrossEntropyLoss()
res=f_loss(torch.Tensor([[-100,10,8]]),torch.LongTensor([1]))
print("CrossEntropyLoss {}".format(res))


optimizer=torch.optim.Adam(model.parameters())

#############train
def train(model,loader,f_loss,optimizer,device):


    model.train()
    for i,(inputs,targets) in enumerate(loader):
        inputs,targets=inputs.to(device),targets.to(device)
        outputs=model(inputs)
        loss=f_loss(outputs,targets)
        #loss
        #tensor(0.5546, device='cuda:0', grad_fn= < NllLossBackward >)
        #loss.shape torch.Size([])

        optimizer.zero_grad()
        loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass
        optimizer.step()

############# test
def test(model,loader,f_loss,device):


    with torch.no_grad():
        model.eval()
        N=0
        tot_loss,correct=0.0,0.0

        for i,(inputs,targets) in enumerate(loader):

            inputs,targets=inputs.to(device),targets.to(device)

            outputs=model(inputs)

            N+=inputs.shape[0]

            tot_loss+=inputs.shape[0]*f_loss(outputs,targets).item()

            predicted_targets=outputs.argmax(dim=1)
            #predicted_targets.shape  torch.Size([128])
            correct+=(predicted_targets==targets).sum().item()

        return tot_loss/N,correct/N


top_logdir="./logs"
if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logdir=generate_unique_logpath(top_logdir,"vanilla")
print("logging to {}".format(logdir))
if not os.path.exists(logdir):
    os.mkdir(logdir)


model_checkpoint=ModelCheckpoint(logdir+"/best_model.pt",model)

epochs = 10

for t in range(epochs):
    print("Epoch {}".format(t))
    train(model, train_loader, f_loss, optimizer, device)

    val_loss, val_acc = test(model, valid_loader, f_loss, device)
    print("Validation: Loss: {:.4f}, Acc: {:.4f}".format(val_loss, val_acc))
    model_checkpoint.update(val_loss)


model_path=logdir+"/best_model.pt" #"./logs/linear_0/best_model.pt"
loaded_dict=torch.load(model_path)
print("loaded:",loaded_dict)
model=loaded_dict.to(device)


model.eval()
test_loss,test_acc=test(model,test_loader,f_loss,device)
print("after load with test : Loss: {:.4f}, Acc: {:.4f}".format(test_loss,test_acc))
