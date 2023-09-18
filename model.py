import torch
import torch.nn as nn
from torch.nn import functional as F

# Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

# Decoder Network
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor):
        h = self.relu(self.fc1(z))
        #x_recon = torch.atan(self.fc2(h))
        out = self.fc2(h)
        return out

# VAE combining the Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    @torch.no_grad()
    def generate(self, num_samples: int):
        # Sample from the standard normal distribution (since VAE's latent space is constrained to be close to this distribution)
        # z ~ N(0,1)
        z = torch.randn(num_samples, self.decoder.fc1.in_features)  # Assuming the latent dimension is the input feature size for the decoder's first layer
        z = z.to(next(self.parameters()).device)
        
        # Decode z to get the data samples
        generated_data = self.decoder(z)
        return generated_data

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var