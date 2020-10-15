
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision.models import resnet50

from efficientnet_pytorch import EfficientNet

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tqdm

from utilities import *

if __name__ == "__main__":
    
    params = {'size_input': 456,
            'workers': 8,
            'batch_size': 16,
            'num_classes':4}


    data_path = 'C:/Users/vcl/Desktop/data/test.csv'
    dataFrame = pd.read_csv(data_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    IDs = dataFrame.image_id.to_list()
                        
    test_dataset = PyTorchTestImageDataset(image_list = IDs, size_image = params['size_input'])
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = params['batch_size'], shuffle = False, num_workers=params['workers'])


    # model = resnet50()
    # model.fc = nn.Linear(model.fc.in_features, params['num_classes'])
    model = EfficientNet.from_pretrained('efficientnet-b6')
    model._fc = nn.Linear(model._fc.in_features,params['num_classes'])

    weights_path = 'C:/Users/vcl/Desktop/efficientnetb6_ver3_epochs_20_loss_0.0116.pth'
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    softmax = nn.Softmax()

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_dataloader, position=0, leave=True)):
            
            data_batch = data.to(device)
                            
            b_size = data_batch.size(0)

            output = model(data_batch)

            prob = softmax(output)
            
            if i == 0:
                torch.cuda.synchronize() 
                valid_prob = prob.cpu().detach().numpy()
            else:
                torch.cuda.synchronize() 
                temp_prob = prob.cpu().detach().numpy()
                valid_prob = np.concatenate((valid_prob, temp_prob), axis = 0)
    # print(valid_prob.shape)
    path = 'C:/Users/vcl/Desktop/data/sample_submission.csv'
    df = pd.read_csv(path)
    valid_prob = np.round(valid_prob,4)
    df.healthy = valid_prob[:,0]
    df.multiple_diseases = valid_prob[:,1]
    df.rust = valid_prob[:,2]
    df.scab = valid_prob[:,3]
    df.to_csv('submission.csv', index=False)