import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from VAE import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse

# configuration

argp = argparse.ArgumentParser()
argp.add_argument('--writing_params_path',default=None)
args = argp.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
hidden_dim = 300
num_epochs = 80
batch_size = 32
lr = 5e-7

# dataset loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = VAE(input_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='sum')

# start training
loss_list = []
for epoch in tqdm(range(num_epochs)):
    pbar = tqdm(enumerate(train_loader))
    for i, (x_batch, _) in pbar:
        x_batch = x_batch.to(device).view(x_batch.shape[0], input_dim)
        x_reconstructed, mu, logvar = model(x_batch)
        reconstruction_loss = loss_fn(x_reconstructed, x_batch)
        kl_divergence = torch.sum(mu - logvar - 1 + torch.exp(0.5 * logvar))

        optimizer.zero_grad()
        loss = reconstruction_loss + kl_divergence
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

#model = model.to("cpu")
torch.save(model, args.writing_params_path)





