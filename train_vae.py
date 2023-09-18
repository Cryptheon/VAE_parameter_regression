import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
# Assuming you have already imported or defined LineDataset and VAE
# If not, you should import or define them before using
from data import LineDataset
from model import VAE

from utils import plot_sampled_points

import torch 
import torch.nn.functional as F

def vae_parameter_loss(output_slopes: torch.Tensor, slopes: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
    # Reconstruction loss
    recon_loss = F.mse_loss(output_slopes, slopes, reduction='mean')
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kl_loss

# Training loop
def train_vae(model: VAE, dataloader: DataLoader, optimizer: torch.optim.Adam, epochs: int=10):
    model.train()
    losses = []
    for epoch in range(epochs):
        for i, (input_tensor, target)  in enumerate(dataloader):
            optimizer.zero_grad()
            input_tensor, target = input_tensor.float(), target.float()
            output_slopes, mu, log_var = model(input_tensor)
            
            recon_loss, kl_loss = vae_parameter_loss(output_slopes, target, mu, log_var)
            loss = recon_loss + kl_loss#*0.001
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if i % 100 == 0:
                losses.append(np.array(losses[-100:]).mean())
                print("kl loss: ", kl_loss.item(), kl_loss.item()*0.001)
                print("recon_loss:", recon_loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.array(losses)[-1000:].mean():.4f}")
    return losses

def generate(model: VAE, Dataset: DataLoader, args):
    model.eval()
    generated_slopes = model.generate(args.generate_num)
    generated_slopes = generated_slopes.cpu().numpy()

    x, lines = Dataset.generate_lines(generated_slopes)
    #print(lines)
    sampled_points = Dataset.sample_points_from_lines(x, lines)

    # Plot the sampled points
    plot_sampled_points(sampled_points)

def main(args):

    dataset = LineDataset(args.num_lines, args.input_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize VAE and optimizer
    model = VAE(args.input_dim*2, args.hidden_dim, args.latent_dim, args.output_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train VAE
    losses = train_vae(model, dataloader, optimizer, epochs=args.epochs)
    plt.plot(losses)
    plt.show()

    generate(model, dataset, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_lines", type=int, default=3000)
    parser.add_argument("--generate_num", type=int, default=64)

    args = parser.parse_args()

    main(args)