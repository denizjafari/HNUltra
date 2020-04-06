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
import json

import wandb


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
parser.add_argument('--layers-dim', nargs='+', type=int, default=[100])
parser.add_argument('--input-dim', type=int, default=100, 
                    help='size of vae latent dimension')

args = parser.parse_args()
print(args)

configs_dict = {'num_epochs':args.epochs,'layers_dim':args.layers_dim, 'vae_latent_dim':args.input_dim}

run_id = 'vae_view_{}_input_dim_{}_layers_'.format(args.input_dim, len(args.layers_dim)) + "_".join(list(map(str, args.layers_dim)))
print("run_id =",run_id)

wandb_username = 'nkorhani'

if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')


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

# root directory
andrea_dir = "/home/andreasabo/Documents/HNProject/"

data_dir = "/home/navidkorhani/Documents/HNProject/HNUltra/latent{}_images/".format(args.input_dim)

# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6}

# read target df
csv_path = os.path.join(andrea_dir, "all_splits_1000000.csv")
data_df = pd.read_csv(csv_path, usecols=['subj_id', 'image_ids', 'view_label', 'view_train'])

label_mapping = {'Other':0, 'Saggital_Right':1, 'Transverse_Right':2, 
                 'Saggital_Left':3, 'Transverse_Left':4, 'Bladder':5}

reverse_mapping = {v:k for k,v in label_mapping.items()}

data_df['view_label'] = data_df['view_label'].map(label_mapping)

def get_test_loader():
    test_df = data_df[data_df.view_train == 0]

    test_ids = []
    test_labels = {}
    for ind, row in test_df.iterrows():
        test_ids.append(row['image_ids'])
        test_labels[row['image_ids']] = row['view_label']

    test_dataset = Dataset(test_ids, test_labels)
    test_loader = data.DataLoader(test_set, **params)

    return test_loader
        
def get_train_and_val_loaders(fold):
    with open('data_splits.json') as f:
        splits = json.load(f)

        train_ids = splits[fold]['train_ids']
        train_labels = splits[fold]['train_labels']

        val_ids = splits[fold]['valid_ids']
        val_labels = splits[fold]['valid_labels']

    # Datasets and Generators
    train_dataset = Dataset(train_ids, train_labels)
    train_loader = DataLoader(train_dataset, **params)

    val_dataset = Dataset(val_ids, val_labels)
    val_loader = data.DataLoader(val_dataset, **params)

    return train_loader, val_loader

class DeepClassifier(nn.Module):
    def __init__(self, input_dim=100, layers_dim = [200]):
        super(DeepClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.layers_dim = layers_dim.copy()

        self.num_of_layers = len(self.layers_dim)

        self.fc1 = nn.Linear(input_dim, self.layers_dim[0])

        if self.num_of_layers == 2:
            self.fc2 = nn.Linear(self.layers_dim[0], self.layers_dim[1])

        if self.num_of_layers == 3:
            self.fc3 = nn.Linear(self.layers_dim[1], self.layers_dim[2])

        if self.num_of_layers == 4:
            self.fc4 = nn.Linear(self.layers_dim[2], self.layers_dim[3])

        if self.num_of_layers == 5:
            self.fc5 = nn.Linear(self.layers_dim[3], self.layers_dim[4])


        self.fc_last.append(nn.Linear(self.layers_dim[-1], 6))
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self, x):

        out = F.relu(self.fc1(x))

        if self.num_of_layers == 2:
            out = F.relu(self.fc1(x))

        if self.num_of_layers == 3:
            out = F.relu(self.fc2(x))

        if self.num_of_layers == 4:
            out = F.relu(self.fc3(x))

        if self.num_of_layers == 5:
            out = F.relu(self.fc4(x))

        out = self.softmax(self.fc_last(out))
        
        return out

def compute_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm.diagonal()


def train(epoch, model, optimizer, criterion, train_loader):
    
    model.train()
    train_loss = 0
    all_targets = np.array([])
    all_preds = np.array([])
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        
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
    
    
 #   train_loss_array = np.append(train_loss_array, avg_train_loss)

    acc = float(np.sum(all_targets == all_preds))/len(all_preds)
    return avg_train_loss, acc, all_targets, all_preds


def evaluation(model, criterion, val_loader):
    #global val_loss_array
    
    all_targets = np.array([])
    all_preds = np.array([])
    
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

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
        
def test(model, criterion, test_loader):

    #global test_loss_array
    
    model.eval()
    test_loss = 0
    
    all_targets = np.array([])
    all_preds = np.array([])
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            preds = model(inputs)
            
            test_loss += criterion(preds, targets).item()

            prob_pred = preds.detach().cpu().numpy() #128 x 6
            pred_label = np.argmax(prob_pred, axis=1)
            
            all_preds = np.append(all_preds, pred_label)
            all_targets  = np.append(all_targets, targets.detach().cpu().numpy())


    test_loss /= len(test_loader.dataset) * len(inputs)
    
    acc = float(np.sum(all_targets == all_preds))/len(all_preds)
    #test_loss_array = np.append(test_loss_array, test_loss)
    
    return test_loss, acc, all_targets, all_preds

if __name__ == "__main__":

    test_loader = get_test_loader()
    for fold in range(5):
       
        f = open(run_id+'_fold_'+str(fold)+'output.txt', 'w')

        wandb.init(project='hnultra', name=run_id+'_fold_'+str(fold)
        wandb.update(config_dict)

        train_loader, val_loader = get_train_and_val_loaders(fold)

        model = DeepClassifier(input_dim=args.input_dim, layers_dim=args.layers_dim).to(device)
        wandb.watch(model)
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        least_error=-1

        val_loss_array = np.array([])

        for epoch in range(args.epochs):

            train_loss, train_acc, train_targets, train_preds = train(epoch, model, optimizer, criterion, train_loader)
            val_loss, val_acc, val_targets, val_preds = evaluation(model, criterion, val_loader)
            
            val_loss_array = np.append(val_loss_array, val_loss)

            if least_error==-1 or least_error>val_loss:
                PATH = os.path.join('saved models', run_id+'.pt')
            
                torch.save(model.state_dict(), PATH)
                least_error = val_loss

            print('====> Epoch: {}  Train loss: {:.2f}   Train Accuracy: {:.2f}%   |  Validation loss: {:.2}   Validation Accuracy: {:.2f}%'.format(
                      epoch, train_loss, train_acc*100, val_loss, val_acc*100))

            wandb.log({'train_loss':train_loss, 'val_loss':val_loss}, step=epoch)

            precision_recall_fscore_support(train_targets, train_preds)
            precision_recall_fscore_support(val_targets, val_preds)


            if epoch >= 40 and val_loss_array[-1] >= min(val_loss_array[-20:-1]):
                print('Early stopping')
                break

        #Load the model with the best performance
        PATH = os.path.join('saved models', run_id+'.pt')
        model.load_state_dict(torch.load(PATH))

        val_loss, val_acc, val_targets, val_preds = evaluation(model, criterion, val_loader)
        print('Best validation accuracy:', val_acc}
        f.write('Best validation accuracy: {.2f} ... best validation loss: {.2f}\n'.format(val_acc, val_loss)

        class_acc = compute_class_accuracy(val_targets, val_preds)
        print("Validation class accuracy:")
        print(class_acc)

        f.write('validation class accuracy:' + str(class_acc) + '\n')

        test_loss, test_acc, test_targets, test_preds= test(model, criterion, test_loader)
        class_acc = compute_class_accuracy(test_targets, test_preds)
        print("Test class accuracy:")
        print(class_acc)

        f.write('test class accuracy:' + str(class_acc) + '\n')


        print("Test Loss: {:.2f}   Test Accuracy: {:.2f}%".format(test_loss, test_acc*100))

        f.write("Test Loss: {:.2f}   Test Accuracy: {:.2f}%\n".format(test_loss, test_acc*100))

        f.close()


    wandb.save(run_id+'.h5')

    model.save(os.path.join(wandb.run.dir,run_id+'h5'))

    


