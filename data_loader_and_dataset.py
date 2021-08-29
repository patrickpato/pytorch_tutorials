import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader, dataloader
import numpy as np 
import math


#implementing custom dataset
class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter = ",", dtype = np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) #all samples(rows) minus the first column
        self.y = torch.from_numpy(xy[:, [0]]) #all samples. n_samples, 1
        self.n_samples=  xy.shape[0]



    def __getitem__(self, index):
        #indexing data
        return self.x[index], self.y[index]
    
    def __len__(self):
        #len(data)
        return self.n_samples
dataset = WineDataset()
first_data = dataset[0]
features, labels =first_data
#print(features), print(labels)

#implementing dataloader
dataloader = DataLoader(dataset = dataset, batch_size=4, shuffle=True, 
num_workers=2)
#training loop
num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        #implement forward and backward pass
        if (1+i) % 5 ==  0:
            print(f'epoch: {epoch+1}/{num_epoch}')
