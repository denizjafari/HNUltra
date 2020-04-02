
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import *
import time
import os
from torch.utils import data
import random


# In[2]:


if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')


# In[3]:


# root directory
andrea_dir = "/home/andreasabo/Documents/HNProject/"

# data directory on current machine: abhishekmoturu, andreasabo, denizjafari, navidkorhani
data_dir = "/home/navidkorhani/Documents/HNProject/HNUltra/latent100_images/"

# read target df
csv_path = os.path.join(andrea_dir, "all_splits_1000000.csv")
data_df = pd.read_csv(csv_path, usecols=['subj_id', 'image_ids', 'view_label', 'view_train'])


# In[4]:


batch_size = 128


# In[5]:


label_mapping = {'Other':0, 'Saggital_Right':1, 'Transverse_Right':2, 
                 'Saggital_Left':3, 'Transverse_Left':4, 'Bladder':5}

data_df['view_label'] = data_df['view_label'].map(label_mapping)

train_df = data_df[data_df.view_train == 1]
test_df = data_df[data_df.view_train == 0]

labels = {}
train_and_valid_subj_ids = []
train_and_valid_image_ids = []
test_ids = []

for ind, row in train_df.iterrows():
    train_and_valid_subj_ids.append(row['subj_id'])
    train_and_valid_image_ids.append(row['image_ids'])
    labels[row['image_ids']] = row['view_label']

for ind, row in test_df.iterrows():
    test_ids.append(row['image_ids'])
    labels[row['image_ids']] = row['view_label']

s = set()
t_v_ids = pd.DataFrame(list(zip(train_and_valid_subj_ids, train_and_valid_image_ids)), columns=['subj_ids', 'image_ids'])
id_groups = [t_v_ids for _, t_v_ids in t_v_ids.groupby('subj_ids')]
random.shuffle(id_groups)
id_groups = pd.concat(id_groups).reset_index(drop=True)
train_val_split = int(0.8*len(set(id_groups['subj_ids'].values)))
train_val_set = [i for i in id_groups['subj_ids'].values if not (i in s or s.add(i))]
cutoff = train_val_set[train_val_split]
train_portion = (id_groups['subj_ids'].values == cutoff).argmax()

train_ids = id_groups[:train_portion]['image_ids'].tolist()
valid_ids = id_groups[train_portion:]['image_ids'].tolist()

partition = {'train':train_ids, 'valid':valid_ids, 'test':test_ids}


# In[6]:


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        path = data_dir + ID + '.npy'
        z = torch.tensor(np.load(path)).squeeze()


        y = torch.tensor(self.labels[ID], dtype=torch.long)

        return z, y


# In[7]:


# Data augmentation and normalization for training

# Parameters
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}

# Generators
training_set = Dataset(partition['train'], labels)
train_loader = DataLoader(training_set, **params)

validation_set = Dataset(partition['valid'], labels)
validation_loader = data.DataLoader(validation_set, **params)

test_set = Dataset(partition['valid'], labels)
test_loader = data.DataLoader(test_set, **params)


# In[13]:


class DeepClassifier(nn.Module):
    def __init__(self):
        super(DeepClassifier, self).__init__()
        
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 6)
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        out = self.softmax(self.fc3(h2))
        
        return out
    


# In[14]:


#hyperparams

epochs = 30

val_check_interval = 40

log_interval = 10


# In[15]:


classifier_model = DeepClassifier().to(device)

optimizer = optim.Adam(classifier_model.parameters(), lr=1e-4)

criterion = nn.CrossEntropyLoss()

train_loss_array = np.array([])
val_loss_array = np.array([])
least_error = -1


# In[16]:


def train(epoch):
    global train_loss_array
    
    classifier_model.train()
    train_loss = 0
    all_targets = np.array([])
    all_preds = np.array([])
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = classifier_model(inputs)
        
        loss = criterion(preds, targets)
        train_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        
        prob_pred = preds.detach().cpu().numpy() #128 x 6
        pred_label = np.argmax(prob_pred, axis=1)
        
        all_preds = np.append(all_preds, pred_label)
        all_targets  = np.append(all_targets, targets.detach().cpu().numpy())
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} '.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
        
        #print(len(train_loader.dataset), len(inputs))    
        avg_train_loss = train_loss / len(train_loader.dataset) * len(inputs)
    
    #print('====> Epoch: {} Average Training loss: {:.2f}'.format(
    #      epoch, avg_train_loss))
    
    train_loss_array = np.append(train_loss_array, avg_train_loss)

    acc = float(np.sum(all_targets == all_preds))/len(all_preds)
    return avg_train_loss, acc


def evaluation():
    global least_error
    global val_loss_array
    
    all_targets = np.array([])
    all_preds = np.array([])
    
    classifier_model.eval()
    eval_loss = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            preds = classifier_model(inputs)

            loss = criterion(preds, targets)
            eval_loss += loss.item()
            
            prob_pred = preds.detach().cpu().numpy() #128 x 6
            pred_label = np.argmax(prob_pred, axis=1)
            
            all_preds = np.append(all_preds, pred_label)
            all_targets  = np.append(all_targets, targets.detach().cpu().numpy())

        avg_val_loss = eval_loss / len(validation_loader.dataset) * len(inputs)
        
    val_loss_array = np.append(val_loss_array, avg_val_loss)
    
    if least_error==-1 or least_error>avg_val_loss:
        PATH = os.path.join('saved models', 'vae_classifier.pt')
    
        torch.save(classifier_model.state_dict(), PATH)
        
        least_error = avg_val_loss
    
    acc = float(np.sum(all_targets == all_preds))/len(all_preds)
    
    return avg_val_loss, acc
        
def test():

    #global test_loss_array
    
    classifier_model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            preds = classifier_model(inputs)
            
            test_loss += criterion(preds, targets).item()

    test_loss /= len(test_loader.dataset) * len(inputs)
    
    #test_loss_array = np.append(test_loss_array, test_loss)
    
    return test_loss


# In[17]:


for epoch in range(epochs):
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = evaluation()
    
    print('====> Epoch: {}  Train loss: {:.2f}   Train Accuracy: {:.2f}%   |  Validation loss: {:.2}   Validation Accuracy: {:.2f}%'.format(
              epoch, train_loss, train_acc, val_loss, val_acc))
    
test_loss = test()

print("Test Loss: {}".format(test_loss))

np.save('vae_class_train.npy', train_loss_array)
np.save('vae_class_val.npy', val_loss_array)


