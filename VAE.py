
# coding: utf-8

# In[1]:

from __future__ import print_function
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import os
from PIL import Image
import argparse




# In[2]:


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


# root directory
root_dir = "/home/andreasabo/Documents/HNProject/"

# data directory on current machine: abhishekmoturu, andreasabo, denizjafari, navidkorhani
data_dir = "/home/navidkorhani/Documents/HNProject/all_label_img/"

# read target df
csv_path = os.path.join(root_dir, "all_splits_100000.csv")
data_df = pd.read_csv(csv_path, usecols=['image_ids', 'view_train'])


# In[4]:


train_df = data_df[data_df.view_train != 0]
test_df = data_df[data_df.view_train == 0]

print("# of images in train = {}, # of images in test = {}".format(len(train_df), len(test_df)))
train_ids = []
test_ids = []

for ind, row in train_df.iterrows():
    train_ids.append(row['image_ids'])

for ind, row in test_df.iterrows():
    test_ids.append(row['image_ids'])

partition = {'train':train_ids, 'test':test_ids}


# In[5]:


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img_path = data_dir + ID + '.jpg'
        image = Image.open(img_path).convert('L')
        image = ToTensor()(image)

        return image


# In[6]:


args.cuda= not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda:1" if args.cuda else "cpu")

print("device is", device)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6}

if args.cuda:
    params['num_workers']=1
    params['pin_memory']=True


# Data Loader
training_set = Dataset(partition['train'])
train_loader = data.DataLoader(training_set, **params)

test_set = Dataset(partition['test'])
test_loader = data.DataLoader(test_set, **params)


# In[7]:


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(65536, 400)
        self.fc21 = nn.Linear(400, 40)
        self.fc22 = nn.Linear(400, 40)
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, 65536)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #print("z.size() =", z.size())
        h3 = F.relu(self.fc3(z))
        #print("h3.size() =", h3.size())
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 65536))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 65536), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #print("data.size()=", data.size())
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    PATH = './vae_model.pt'
    torch.save(model.state_dict(), PATH)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 256, 256)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# In[8]:


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 40).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 256, 256),
                       'results/sample_' + str(epoch) + '.png')

