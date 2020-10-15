import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50

import os
import sys
# sys.path.append(r'C:\Users\emage\OneDrive\Desktop\SIIM-ISIC-Melanoma-Classification\efficientnet_pytorch/')
# sys.path.append('/datapath/')
import glob
import tqdm

from tabulate import tabulate
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
# from model import EfficientNet
from utilities import *

'''
Efficientnet input size:
    + B0: 224
    + B1: 240
    + B2: 260
    + B3: 300
    + B4: 380
    + B5: 456
    + B6: 528
    + B7: 600
'''


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {'workers': 8,
             'batch_size': 1,
             'num_epochs': 20,
             'lr': 0.01,
             'size_image': 600,
             'num_classes':4,
             'checkpoint': True,
             'save_path': './checkpoint/'}

    checkDir(params['save_path'])
    # dataframe = pd.read_csv('./../data/train.csv')
    df = pd.read_csv('C:/Users/vcl/Desktop/data/train.csv')
    
    label = []
    a_df = np.array(df)

    for line in a_df:
        label.append(np.where(line==1)[0][0]-1)

    name = df.image_id
        
    boxes = {'image_id': name,
             'labels': label}

    dataFrame = pd.DataFrame(boxes)

    IDs = dataFrame.image_id.to_list()
    labels = dataFrame.labels.to_list()
    train ,val, y_train,y_val = train_test_split(IDs, labels, test_size = 0.2, random_state = 42, shuffle = True)

    # train, y_train = oversamplingData(X = train, y =y_train)

    train_dataset = PyTorchImageDataset(image_list = train, labels = y_train, \
        train = True, size_image = params['size_image'], hard = False)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = params['batch_size'], \
        shuffle = True, num_workers=params['workers'])

    val_dataset = PyTorchImageDataset(image_list=val, labels=y_val, \
        train = False, size_image = params['size_image'], hard = False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=params['batch_size'], \
        shuffle=True, num_workers=params['workers'])

    dataloader = {'train': train_dataloader, 'valid': val_dataloader}

    '''
    EfficientNet part
    '''
    # weight_path = os.getcwd() + '\\..\\efficientnet_pytorch\\weights\\efficientnet-b5-b6417697.pth'
    # model = effNet.initModel().to(device)
    
    # input size of b5 456 x 456 
    model = EfficientNet.from_pretrained('efficientnet-b7')
     
    # effNet = TobyNet('efficientnet-b0', weights_path = weight_path, num_classes = 2)
    model._fc = nn.Linear(model._fc.in_features, params['num_classes'])
    # weights_path = '/datapath/efficientnetb0_epochs_5_loss_3.0023.pth'
    # model.load_state_dict(torch.load(weights_path))

    '''
    ResNet50 part
    '''
    # model = resnet50()
    # model.fc = nn.Linear(model.fc.in_features,2)

    # weights_path = '/datapath/resnet50_epochs_5_loss_1.6105.pth'
    # model.load_state_dict(torch.load(weights_path))
    
    model.to(device)

    # class_weight = torch.FloatTensor([0.77683887, 1.40305238]).to(device)

    # criterion = nn.CrossEntropyLoss(weight= class_weight)
    criterion = nn.CrossEntropyLoss()

    softmax = nn.Softmax()
    
    # optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999))
    optimizer = optim.SGD(model.parameters(), lr = params['lr'], momentum= 0.9, nesterov= True)
    loss_train = []
    loss_valid = []

    pre_acc = 0.3    


    n_class0, n_class1, n_class2, n_class3 = extractLabelsNums(y_train)
    print(tabulate([['Healthy', n_class0], ['Multiple', n_class1], ['Rust', n_class2], ['Scab', n_class3]], headers=['Name', 'Number'], tablefmt='orgtbl'))

    for epoch in range(params['num_epochs']):
        for phase in ['train','valid']:
            # if (epoch in range(3)) and phase == 'train':
                # freezeResNetFC(model)
            if phase == 'train':
                # unfreezeResNetFC(model)
            #     model.train(True)
            # else:
            #     # model.train(False)
            #     # model.eval()
            #     freezeResNetFC(model)
  
                for i, data in enumerate(tqdm.tqdm(dataloader[phase]), 0):
                    optimizer.zero_grad()
                    data_batch = data[0].to(device)

                    b_size = data_batch.size(0)
                    
                    label = data[1].type(torch.long)

                    label = label.to(device)

                    # torch.save(data_batch, 'test.pt')

                    output = model(data_batch)

                    prob = softmax(output)
                    
                    loss = criterion(output, label)
                    
                    
                    # if phase == 'train':
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()  
                    # print('epoach %d train loss: %.3f' %(epoch + 1, loss.item()))
            else:
                with torch.no_grad():
                    for i, data in enumerate(tqdm.tqdm(dataloader[phase]), 0):
                      data_batch = data[0].to(device)

                      b_size = data_batch.size(0)
                      
                      label = data[1].type(torch.long)

                      label = label.to(device)

                      # torch.save(data_batch, 'test.pt')
                      output = model(data_batch)

                      prob = softmax(output)
                      
                      loss = criterion(output, label)
                      
                      if phase == 'valid' and i == 0:
                        valid_label = label.cpu().detach().numpy()
                        
                        valid_prob = prob.cpu().detach().numpy()
                                                    
                        loss_valid.append(loss.item())
                      else:
                        torch.cuda.synchronize() 

                        temp_label = label.cpu().detach().numpy()

                        valid_label = np.concatenate((valid_label, temp_label))

                        torch.cuda.synchronize() 

                        temp_prob = prob.cpu().detach().numpy()

                        valid_prob = np.concatenate((valid_prob, temp_prob), axis = 0)

                        loss_valid.append(loss.item())
        if phase == 'valid':
            '''
            TODO: 
            + dump loss, auc, recall, precision
            '''
            last = 0

            torch.cuda.synchronize() 
            
            # auc = evaluate(valid_label, valid_prob, save_path = params['save_path'] + 'auc_efficientnetb0_ver3_epochs_' + str(last + epoch+1) + '.png')
            # confusion_matrix(valid_label, valid_prob, params['save_path'] +'efficientnetb0_ver3_epochs_' + str(last + epoch+1) + '.png')

            # auc_list.append(auc)
            
            # if auc > highest_auc:
            #     hightest_auc = auc
            #     save = True

            predict_labels = np.argmax(valid_prob, axis=1)
            acc = accuracy(valid_label, predict_labels)
            class_name = ['Healthy','Both','Rust','Scab']
            confusion_matrix(valid_label, predict_labels, class_name, params['save_path']+str(last + epoch+1))
            print('epoch %d valid acc: %.3f' %(epoch + 1, acc))
            print('epoch %d valid loss: %.3f' %(epoch + 1, loss.item()))
            # print('epoch %d valid auc: %.3f' %(epoch + 1, auc))
            
            
            # if params['checkpoint'] and save:
                # path_name = 'efficentnetB5_epochs_' + str(epoch+1) +'_loss_' + loss.item() + '.pth'
            path_name = params['save_path'] + 'efficientnetb7_epochs_' + str(last + epoch+1) +'_loss_' + str(round(loss.item(),4)) + '.pth'
            if acc > pre_acc:
                torch.save(model.state_dict(), path_name)
                pre_acc = acc

    # PATH = params['save_path'] + 'resnet50.pth'
    # torch.save(model.state_dict(), PATH)            
if __name__ == "__main__":
    train()


