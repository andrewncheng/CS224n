import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_dim + self.hidden_dim),  # one for mean and one for variance
            # nn.ReLU() ReLU only allows for non-negative values. We want to allow mu and sigma to also be negative
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            # if we normalize our images to values between 0-1 then we should apply sigmoid to map values to 0-1
            nn.Sigmoid()
        )

    def encode(self, x):
        # q_\phi(z|x) --> returns mean and variance
        mu, logvar = torch.chunk(self.encoder(x), chunks=2, dim=1)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y

    def sample(self, mu, logvar):
        # given \mu and \var, sample z ~ N(mu, var^2) using trick -- z  = mu + var * epsilon
        # log trick for variance --> std = exp(log( std**2 / 2)) which allows for negatives values of sigma
        print("enter")
        epsilon = torch.normal(torch.zeros(self.hidden_dim), torch.ones(self.hidden_dim))
        z = mu + torch.exp(0.5 * logvar)*epsilon
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.to(self.device)
        logvar = logvar.to(self.device)
        z = self.sample(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar



# print("hello")
# image = torch.randn(4, 28*28)
# vae = VAE(input_dim=784, hidden_dim=256)
# x_reconstruct, mean, sigma = vae(image)
# print(x_reconstruct.shape)
# print(mean.shape)
# print(sigma.shape)




