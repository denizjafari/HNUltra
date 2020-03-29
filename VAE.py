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
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--h-size', type=int, default=400, metavar='N',
                    help='size of the hidden layer')
parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                    help='size of the hidden layer')
args = parser.parse_args()

print(args)

# root directory
root_dir = "/home/andreasabo/Documents/HNProject/"

# data directory on current machine: abhishekmoturu, andreasabo, denizjafari, navidkorhani
data_dir = "/home/navidkorhani/Documents/HNProject/all_label_img/"

# read target df
csv_path = os.path.join(root_dir, "all_splits_100000.csv")
data_df = pd.read_csv(csv_path, usecols=['image_ids', 'view_train'])


train_df = data_df[data_df.view_train != 0]
test_df = data_df[data_df.view_train == 0]

print("# of images in train = {}, # of images in test = {}".format(len(train_df), len(test_df)))
train_ids = []
test_ids = []
least_error = -1
for ind, row in train_df.iterrows():
    train_ids.append(row['image_ids'])

for ind, row in test_df.iterrows():
    test_ids.append(row['image_ids'])

partition = {'train':train_ids, 'test':test_ids}

results_sub_dir = 'h{0}_l{1}_e{2}'.format(args.h_size, args.latent_size, args.epochs)
output_dir = os.path.join('./results', results_sub_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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



args.cuda= not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

print("device is", device)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6}

if args.cuda:
    params['num_workers']=1
    params['pin_memory']=True

print(params)

# Data Loader
training_set = Dataset(partition['train'])
train_loader = data.DataLoader(training_set, **params)

test_set = Dataset(partition['test'])
test_loader = data.DataLoader(test_set, **params)


# In[7]:


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        hidden_dim = args.h_size
        latent_dim = args.latent_size
        self.fc1 = nn.Linear(65536, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 65536)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 65536))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)

optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

batches_loss_array = np.array([])
epochs_loss_array = np.array([])
test_loss_array = np.array([])

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    try:    
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 65536), reduction='sum')
        #BCE = F.mse_loss(recon_x, x.view(-1, 65536), reduction='sum')
    except:
        print("shape is:", recon_x.shape, x.shape)
        #np.save('fail_recon_x.npy', recon_x.detach().cpu().numpy())
        #np.save('fail_x.npy', x.cpu().numpy()) 
        raise Exception("We got here")
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 

    return (BCE + KLD) / len(x)


def train(epoch):
    global batches_loss_array
    global epochs_loss_array
    global least_error
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #print('next batch')
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        if torch.isnan(recon_batch).any().item():
            print("got a nan array")
            print(recon_batch)
            print(mu)
            print(logvar)
            raise Exception('nan value tensor')
            
            
        np.save('fail_recon_x.npy', recon_batch.detach().cpu().numpy())
        np.save('fail_x.npy', data.cpu().numpy())
        
        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()
        loss.backward()
        
        #model_params = filter(lambda p: p.requires_grad, model.parameters())
        #grad_vector = np.concatenate([p.grad.cpu().numpy().flatten() for p in model_params])

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
            batches_loss_array = np.append(batches_loss_array,loss.item())
            
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    print('====> Epoch: {} Average Training loss: {:.2f}'.format(
          epoch, avg_train_loss))
    epochs_loss_array = np.append(epochs_loss_array, avg_train_loss)
    
    if least_error==-1 or least_error>avg_train_loss:
        PATH = os.path.join(output_dir, 'vae_model.pt')
    
        torch.save(model.state_dict(), PATH)
        
        least_error = avg_train_loss


def test(epoch):

    global test_loss_array
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
                         output_dir+'/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_loss_array = np.append(test_loss_array, test_loss)



if __name__ == "__main__":
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    params_count = sum([np.prod(p.size()) for p in model_params])
    print("total number of params = ", params_count)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, args.latent_size).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 256, 256),
                       output_dir+'/sample_' + str(epoch) + '.png')

    plt.plot(batches_loss_array)
    plt.xlabel('batch_number')
    plt.ylabel('loss')
    plt.savefig(output_dir+'/batches_loss_plot.png')

    plt.clf()

    plt.plot(np.log(batches_loss_array))
    plt.xlabel('batch_number')
    plt.ylabel('log loss')
    plt.savefig(output_dir+'/batches_log_loss_plot.png')
    
    plt.clf()

    plt.plot(np.log(epochs_loss_array))
    plt.xlabel('epochs_number')
    plt.ylabel('log loss')
    plt.savefig(output_dir+'/epochs_log_loss_plot.png')
    
    plt.clf()

    plt.plot(np.log(test_loss_array))
    plt.xlabel('epochs_number')
    plt.ylabel('test log loss')
    plt.savefig(output_dir+'/test_log_loss_plot.png')
    
    np.save(output_dir+'batch_loss_array.npy', batches_loss_array)
    np.save(output_dir+'epoch_loss_array.npy', epochs_loss_array)
    np.save(output_dir+'test_loss_array.npy', test_loss_array)
