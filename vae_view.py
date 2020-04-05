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
import argparse

import wandb
wandb.init(project='hnultra')


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')

# root directory
andrea_dir = "/home/andreasabo/Documents/HNProject/"

# data directory on current machine: abhishekmoturu, andreasabo, denizjafari, navidkorhani
data_dir = "/home/navidkorhani/Documents/HNProject/HNUltra/latent100_images/"

# read target df
csv_path = os.path.join(andrea_dir, "all_splits_1000000.csv")
data_df = pd.read_csv(csv_path, usecols=['subj_id', 'image_ids', 'view_label', 'view_train'])

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
params = {'batch_size': args.batch_size,
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
        
        self.fc1 = nn.Linear(100, 800)
        self.fc2 = nn.Linear(800, 100)
        self.fc3 = nn.Linear(100, 6)
        #self.fc4 = nn.Linear(50, 6)
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self, z):
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        #out = F.relu(self.fc3(out))
        out = self.softmax(self.fc3(out))
        
        return out


# In[15]:


classifier_model = DeepClassifier().to(device)

wandb.watch(model)

optimizer = optim.Adam(classifier_model.parameters(), weight_decay=1e-4, lr=1e-4)

criterion = nn.CrossEntropyLoss()

train_loss_array = np.array([])
val_loss_array = np.array([])

# In[16]:

def l2_loss():
    reg_coeff = 0.1
    l2_reg = None
    for p in classifier_model.parameters():
        if l2_reg is None:
            l2_reg = reg_coeff*p.norm(2)
        else:
            l2_reg = l2_reg +reg_coeff*p.norm(2)
    
    return l2_reg
        
def compute_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm.diagonal()


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
        
        loss = criterion(preds, targets)# + l2_loss()
        train_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        
        prob_pred = preds.detach().cpu().numpy() #128 x 6
        pred_label = np.argmax(prob_pred, axis=1)
        
        all_preds = np.append(all_preds, pred_label)
        all_targets  = np.append(all_targets, targets.detach().cpu().numpy())
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} '.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
        
        avg_train_loss = train_loss / len(train_loader.dataset) * len(inputs)
    
    
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
    
    
    acc = float(np.sum(all_targets == all_preds))/len(all_preds)
    
    return avg_val_loss, acc, all_targets, all_preds
        
def test():

    #global test_loss_array
    
    classifier_model.eval()
    test_loss = 0
    
    all_targets = np.array([])
    all_preds = np.array([])
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            preds = classifier_model(inputs)
            
            test_loss += criterion(preds, targets).item()

            prob_pred = preds.detach().cpu().numpy() #128 x 6
            pred_label = np.argmax(prob_pred, axis=1)
            
            all_preds = np.append(all_preds, pred_label)
            all_targets  = np.append(all_targets, targets.detach().cpu().numpy())


    test_loss /= len(test_loader.dataset) * len(inputs)
    
    acc = float(np.sum(all_targets == all_preds))/len(all_preds)
    #test_loss_array = np.append(test_loss_array, test_loss)
    
    return test_loss, acc, all_targets, all_preds


# In[17]:

if __name__ == "__main__":
    least_error=-1
    for epoch in range(args.epochs):

        train_loss, train_acc = train(epoch)
        val_loss, val_acc, val_targets, val_preds = evaluation()
        
        if least_error==-1 or least_error>val_loss:
            PATH = os.path.join('saved models', 'vae_classifier.pt')
        
            torch.save(classifier_model.state_dict(), PATH)
            least_error = val_loss

        print('====> Epoch: {}  Train loss: {:.2f}   Train Accuracy: {:.2f}%   |  Validation loss: {:.2}   Validation Accuracy: {:.2f}%'.format(
                  epoch, train_loss, train_acc*100, val_loss, val_acc*100))
        
    class_acc = compute_class_accuracy(val_targets, val_preds)
    print("Validation class accuracy:")
    print(class_acc)

    test_loss, test_acc, test_targets, test_preds= test()
    class_acc = compute_class_accuracy(test_targets, test_preds)
    print("Test class accuracy:")
    print(class_acc)

    print("Test Loss: {:.2f}   Test Accuracy: {:.2f}%".format(test_loss, test_acc*100))

    np.savez('vae_classification_loss.npz', train_loss=train_loss_array, val_loss=val_loss_array)


