import os
import sys
import time
import math
import datetime
import numpy as np
import scipy.signal
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader,Dataset
# from torchvision import transforms
from tensorboardX import SummaryWriter
from pytorchtools import EarlyStopping
import argparse
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


'''
from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
'''
np.set_printoptions(linewidth=200)
parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("-m","--mode", choices=['train', 'test'], default="train", help="working mode")
parser.add_argument("-f", "--fft", default=False, action="store_true",help="frequency domain as input")
parser.add_argument("-e", "--eli_dc", default=False, action="store_true",help="eliminate DC")
parser.add_argument("-s", "--sub_mean", default=False, action="store_true",help="substract mean value")
args = parser.parse_args()

sub_list_train = [1]
#ges_list_train = [5,6,7,9,10,11,12]
ges_list_train = list(range(1,5))
tri_list_train = list(range(1,10))

#sub_list_valid = [2]
#ges_list_valid = [5,6,7,9,10,11,12]
#ges_list_valid = list(range(5,13))
tri_list_valid = [10]

sub_list_test = [1]
ges_list_test = [1,2,3,4]
tri_list_test = [10]

# Hyper parameters
num_epochs = 100
num_classes = len(ges_list_train)
batch_size = 256
learning_rate = 0.00002
#learning_rate = 0.001
channels_num = 8

VOTE_SIZE = 1 #it's also test batch
VOTE_INTERNAL = 20

Datasets = torch.Tensor(np.load('Dataset.npy'))
print(Datasets.shape)
valid_mat = torch.Tensor(np.load('valid_mat.npy'))
total_data = int(torch.sum(valid_mat).item())
print(total_data)


'''
#testify fft
sample = Datasets[0,0,0,0,0,:]
scio.savemat('sample1.mat',{'test':sample.numpy()})
print(sample)
sample = torch.rfft(sample,1)
sample = torch.irfft(sample,1,signal_sizes=(Datasets.shape[-1],))
scio.savemat('sample2.mat',{'test':sample.numpy()})
#sample = torch.norm(sample,p=2,dim=1)
print(sample)
'''


class EMGData(Dataset): #继承Dataset
    def __init__(self,sub,ges,tri): #__init__是初始化该类的一些基础参数
        sub_list = [i -1 for i in sub] 
        ges_list = [i -1 for i in ges] 
        tri_list = [i -1 for i in tri] 
        #print(sub_list,ges_list,tri_list)
        dataset = Datasets[sub_list]
        dataset = dataset[:,ges_list]
        dataset = dataset[:,:,tri_list]

        self.dataset = torch.Tensor(0)
        self.label = np.empty(0,dtype='int64')
        for i in range(len(ges_list)):
            one_gesture = dataset[:,i,:,:,:,:]
            print(one_gesture.shape)
            count = 0
            for j in range(len(sub_list)):
                for k in range(len(tri_list)):
                    valid_len = int(valid_mat[sub_list[j],ges_list[i],tri_list[k]].item()) #global valid length
                    self.dataset = torch.cat((self.dataset,one_gesture[j,k,0:valid_len]),0)
                    print(self.dataset.shape)
                    count = count + valid_len
            print('Gesture ', i , 'len' , count)
            label_one_gesture = np.ones(count,dtype='int64')*i 
            self.label = np.concatenate((self.label,label_one_gesture),0) 
        

        print('----- > emg tensor size')
        print(self.dataset.shape)
        if args.fft == True:
            self.dataset = torch.rfft(self.dataset,1)
            self.dataset = self.dataset[:,:,1:,:] #eliminate dc 
            self.dataset = torch.norm(self.dataset,p=2,dim=3)
        if args.eli_dc == True: #mutex with fft
            scio.savemat('sample_1.mat',{'test1':self.dataset[0,0,:].numpy()})
            self.dataset = torch.rfft(self.dataset,1)
            print(self.dataset.shape)
            self.dataset[:,:,0,:] = 0 #eliminate dc and recover to time domain
            self.dataset = torch.irfft(self.dataset,1,signal_sizes=(Datasets.shape[-1],))
            print(self.dataset.shape)
            scio.savemat('sample_2.mat',{'test2':self.dataset[0,0,:].numpy()})
        self.len = self.dataset.shape[0]

    def __len__(self):#返回整个数据集的大小
        return self.len
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        
        label = self.label[index]
        data = self.dataset[index] 
        return data,label

# Convolutional neural network part of VGGNet
class FIXED_TICNN_AT(nn.Module):
    def __init__(self,used_gpu, nclass, dropout, output_weight=False, init_weights=False):
        super(FIXED_TICNN_AT, self).__init__()
        self.nclass = nclass
        self.output_weight = output_weight
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim = -1)

        self.pooling = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)


        self.bn_16 = nn.BatchNorm1d(16)
        self.bn_32 = nn.BatchNorm1d(32)
        self.bn_64_3 = nn.BatchNorm1d(64)
        self.bn_64_4 = nn.BatchNorm1d(64)
        self.bn_64_5 = nn.BatchNorm1d(64)
        self.bn_128 = nn.BatchNorm1d(128)
        self.bn_256 = nn.BatchNorm1d(256)
        self.bn_512 = nn.BatchNorm1d(512)

        self.layer_8_64 = nn.Conv1d(channels_num, 64, kernel_size=3, padding=1)
        self.layer_8_16 = nn.Conv1d(channels_num, 16, kernel_size=3, padding=1)
        self.layer_16_32 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.layer_32_64 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.layer_64_64 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.layer_64_128 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.layer_128_128 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.layer_128_256 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.layer_256_256 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.layer_256_512 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.layer_512_512 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.layer_256_256_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_2 = nn.BatchNorm1d(256)
        self.layer_256_256_3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_3 = nn.BatchNorm1d(256)
        self.layer_256_256_4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_4 = nn.BatchNorm1d(256)
        self.layer_256_512_5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn_512_5 = nn.BatchNorm1d(512)
        self.layer_512_512_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn_512_6 = nn.BatchNorm1d(512)
        
        self.classifier2 = nn.Sequential(nn.Linear(64, nclass))


    def forward(self, x):

        # Covolutions

        x = self.layer_8_16(x)
        x = self.bn_16(x)
        x = self.relu(x)
        x = self.layer_16_32(x)
        x = self.bn_32(x)
        x = self.relu(x)
        x = self.layer_32_64(x)
        x = self.bn_64_3(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.layer_64_64(x)
        x = self.bn_64_4(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.layer_64_64(x)
        #x = self.dropout(x)
        x = self.bn_64_5(x)
        x = self.relu(x)

        '''
        x = self.layer_64_128(x)
        x = self.bn_128(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.layer_128_128(x)
        x = self.bn_128(x)
        x = self.relu(x)
        x = self.layer_128_128(x)
        x = self.bn_128(x)
        x = self.relu(x)
        '''

        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        #x = self.dropout(x)
        x = self.classifier2(x)
        return x

# Instantiation function of VGG-16 Network with Batch Normalization
def fixedticnn16at_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    #model = FIXED_TICNN_AT(make_layers(cfg['N'], batch_norm=True), **kwargs)
    model = FIXED_TICNN_AT(**kwargs)
    return model

#def show_command_description(args):

#Data Loader

if args.mode == 'train':
    data_train = EMGData(sub_list_train,ges_list_train,tri_list_train)

    dataloader_train = DataLoader(data_train,batch_size=batch_size,shuffle=True)#使用DataLoader加载数据
    print('----> train data ready')

    print('----> validation data begin')
    data_val = EMGData(sub_list_train,ges_list_train,tri_list_valid)

    dataloader_val = DataLoader(data_val,batch_size=batch_size,shuffle=False)#使用DataLoader加载数据
    print('----> validation data ready')

    time = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    log_name = 'runs/'+time + '_' + str(Datasets.shape[-1])
    if args.fft == True:
        log_name = log_name + '_fft'
    if args.eli_dc == True:
        log_name = log_name + '_edc'
    writer = SummaryWriter(logdir=log_name)
    print('----> train data begin')

else:
    print('----> test data begin')
    data_test = EMGData(sub_list_test,ges_list_test,tri_list_test)

    dataloader_test = DataLoader(data_test,batch_size=VOTE_SIZE,shuffle=False)#使用DataLoader加载数据
    print('----> test data ready')
    total_test = data_test.__len__() / VOTE_SIZE

#model

if torch.cuda.is_available():
#if 0:
    device = 'cuda:0'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = fixedticnn16at_bn(used_gpu=[0],nclass=num_classes,dropout=0.5)
    model = torch.nn.parallel.DataParallel(model, device_ids=[0])

else:
    print('use cpu')
    device = 'cpu'
    model = fixedticnn16at_bn(used_gpu=None,nclass=num_classes,dropout=0.5)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-1)

if args.mode == 'train':
    # Train the model
    total_step = len(dataloader_train)
    loss_list = []
    loss_acc_pre = 0
    correct_pre = 0
    early_stopping = EarlyStopping(patience=30, verbose=True)

    print('Training...')
    for epoch in range(num_epochs):
        model.train()
        loss_acc = 0
        correct = 0
        total = 0
        for i, (images,labels) in enumerate(dataloader_train):
            #print('---->' + images + ':' + labels)
            images = images.to(device)
            labels = labels.to(device)
            #if i == 0:
            #    print(labels)
            # Forward pass
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss_acc += loss.item() 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Train      Epoch [{}/{}], Step [{}/{}], Loss: {:.7f},Time:{}' .format(epoch+1, num_epochs, i+1, total_step, loss_acc,datetime.datetime.now()))
        print(correct,",",correct/total)
        writer.add_scalar('train', loss_acc, global_step=epoch)
        loss_list.append(loss.item())

        model.eval()
        with torch.no_grad():
            loss_acc = 0
            correct_valid = 0
            total = 0
            correct_list = np.zeros(num_classes)
            confusion_mat = np.zeros((num_classes,num_classes))
            for i, (images,labels) in enumerate(dataloader_val):
                #print('---->' + images + ':' + labels)
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_acc += loss.item() 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                correct_label = (predicted == labels)
                for j in range(num_classes):
                    correct_ele = (predicted == j) # predict value
                    correct_list[j] = correct_list[j] + (correct_label & correct_ele).sum().item()#(predicted == i).sum().item()
                    
                    #display confusion matrix 
                    label_ele = (labels == j) # predict value
                    for k in range(num_classes):
                        predict_ele = (predicted == k)
                        confusion_mat[j][k] = confusion_mat[j][k] + (label_ele & predict_ele).sum().item()

            print ('Validation Epoch [{}/{}], Step [{}/{}], Loss: {:.7f},Time:{}' .format(epoch+1, num_epochs, i+1, total_step, loss_acc,datetime.datetime.now()))
            print(correct_valid,",",correct_valid/total)
            print(correct_list)
            print(confusion_mat)
            writer.add_scalar('validation', loss_acc, global_step=epoch)
            writer.add_scalar('validation_acc', correct_valid/total, global_step=epoch)

        
        #early_stopping(loss_acc, model)
        
        #if early_stopping.early_stop:
        '''
        stop = True
        for i in range(4):
            stop = stop and ((correct_list[i] / (total/4)) > 0.51)
        if stop == True:
            print("Satisfy Early stopping")
            if loss_acc_pre == 0:
                print("first time")
                loss_acc_pre = loss_acc
                correct_pre = correct_valid
                torch.save(model.state_dict(), 'model.ckpt')
            elif loss_acc < loss_acc_pre or correct_valid < correct_pre:
                print("continue update:",loss_acc_pre,"->",loss_acc)
                if loss_acc < loss_acc_pre:
                    loss_acc_pre = loss_acc
                if correct_valid < correct_pre:
                    correct_valid = correct_pre
                torch.save(model.state_dict(), 'model.ckpt')
            else:
                break
        elif loss_acc_pre != 0:
            break
        '''

        for p in optimizer.param_groups:
            lr = (1 + math.cos(epoch*math.pi/num_epochs)) * learning_rate / 2
            print("lr------>:",lr)
            p['lr'] = lr
        '''
        if epoch == 35:
            for p in optimizer.param_groups:
                print('1111')
                p['lr'] *= 0.1

        if epoch == 80:
            for p in optimizer.param_groups:
                print('1111')
                p['lr'] *= 0.1
        if epoch == 120:
            for p in optimizer.param_groups:
                print('1111')
                p['lr'] *= 0.1
        '''

    # Save the model checkpoint
    #if stop != True and loss_acc_pre != 0:
    torch.save(model.state_dict(), 'model.ckpt')
    #torch.save(torch.load('checkpoint.pt'), 'model.ckpt')
    writer.close()
else:
    #Load model
    model.load_state_dict(torch.load('model.ckpt'))

    acc_list = []
    label_list = []
    predict_list = []
    print('Testing...')
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
                
        correct = 0
        total = 0
        correct_list = np.zeros(num_classes)
        confusion_mat = np.zeros((num_classes,num_classes))
        tic = datetime.datetime.now()
        for images,labels in dataloader_test:
            vote_barrel = np.zeros(num_classes)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            label_list = label_list + (labels.cpu().numpy().tolist())
            predict_list = predict_list + (predicted.cpu().numpy().tolist())
            print(labels)
            print(predicted)
            for i in range(labels.size(0)):
                vote_barrel[predicted[i]] = vote_barrel[predicted[i]] + 1
                #confusion_mat[labels[i]][predicted[i]] = confusion_mat[labels[i]][predicted[i]] + 1
            vote_result = vote_barrel.tolist().index(max(vote_barrel))
            print(vote_barrel)
            print(vote_result)
            total += 1
            if vote_result == labels[0]:
                correct = correct + 1
            #correct_label = (predicted == labels)
            #for i in range(8):
            #    correct_ele = (predicted == i)
            #    correct_list[i] = correct_list[i] + (correct_label & correct_ele).sum().item()#(predicted == i).sum().item()
                #for j in range(8):
                #    correct_list[i] = correct_list[i] + (correct_label & correct_ele).sum().item()#(predicted == i).sum().item()
        #if epoch % 5 == 0:
        #    acc = 100 * correct / total 
        #    acc_list.append(acc)
        tac = datetime.datetime.now()
        print(correct,total,total_test)
        delta = (tac - tic)
        elapse = delta.seconds*1000000 + delta.microseconds
        print(elapse,elapse/total_test)
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        #print(correct_list)
        #for i in range(8):
        #    print('Label {} Accuracy:{} %'.format(i,100*correct_list[i]/(total/8)))
        #print(confusion_mat)
        sns.set()
        f,ax=plt.subplots()
        C2= confusion_matrix(label_list, predict_list, labels=list(range(num_classes)))
        sns.heatmap(C2,annot=True,ax=ax,fmt='d') #画热力图

        ax.set_title('confusion matrix') #标题
        ax.set_xlabel('predict') #x轴
        ax.set_ylabel('true') #y轴
        plt.show()

