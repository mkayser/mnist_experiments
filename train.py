import torch
from torchvision import datasets, transforms
from torch import nn, optim
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import models.autoencoders
import models.discriminators
import analysis.visualization


class LowPassFilter:
    def __init__(self, cutoff_freq, shape=(28,28)):
        self.cutoff = cutoff_freq
        def distance_from_corner(i,j):
            distx = np.minimum(i, shape[0]-i)
            disty = np.minimum(j, shape[1]-j)
            return np.sqrt(distx**2 + disty**2)
        distance_array = np.fromfunction(distance_from_corner, shape, dtype=np.float32)
        self.filter = (distance_array < self.cutoff).astype(np.float32)
        
    def __call__(self, img):
        img = img.numpy()
        assert(np.all( (img>=0.0) & (img<=1.0) ))
        F = np.fft.fft2(img, axes=(1,2))     # shape (1,28,28), complex
        F_filtered = self.filter * F
        ifft = np.fft.ifft2(F_filtered, axes=(1,2))    # back to image domain, still complex
        recon_clipped = np.clip(np.real(ifft), 0.0, 1.0)
        assert(recon_clipped.shape == img.shape)
        assert(np.all( (recon_clipped>=0.0) & (recon_clipped<=1.0) ))
        
        return torch.from_numpy(recon_clipped)



def evaluate_model(model, testloader):
    # Evaluation loop
    model.eval()
    total_loss = 0
    total = 0

    rec_criterion = nn.BCELoss(reduction='sum')

    all_cpu_np_images = []
    all_cpu_np_outputs = []

    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            outputs, mu, logvar, latentz = model(images)
            recon_losses = rec_criterion(outputs, images)
            total += images.size(0)
            total_loss += recon_losses.sum()
            
            cpu_np_images = images.cpu().numpy()
            cpu_np_outputs = outputs.detach().cpu().numpy()
            
            all_cpu_np_images.append(cpu_np_images)
            all_cpu_np_outputs.append(cpu_np_outputs)


    average_loss = total_loss / total
    print(f"\nAverage Loss: {average_loss:.3f}%")

    # Concatenate all test outputs
    images_tensor = np.concatenate(all_cpu_np_images, axis=0)
    outputs_tensor = np.concatenate(all_cpu_np_outputs, axis=0)

    # Select a random subset
    N = images_tensor.shape[0]
    idx = np.random.permutation(N)[:24]
    images_tensor = images_tensor[idx]
    outputs_tensor = outputs_tensor[idx]

    # For the purpose of image display, we eliminate the channels axis
    images_tensor = images_tensor.squeeze(1)  # now (N,28,28)
    outputs_tensor = outputs_tensor.squeeze(1)  # now (N,28,28)

    #analysis.visualization.display_mnist_inputs_and_outputs(images_tensor, outputs_tensor, dims=(None,10))


def train_VAE(
        model,
        trainloader,
        n_epochs,
        optimizer,
        dmodel = None,
        doptimizer = None,
        kl_min_loss_per_sample_per_dim = 0.1,
        kl_loss_schedule = lambda epoch: min(epoch * 0.1, 0.5),
        latentdiscrim_loss_schedule = lambda epoch: min(epoch * 0.1, 0.5)):

    rec_criterion = nn.BCELoss(reduction='sum')

    run_discrim_model = True
    if (dmodel is None and doptimizer is None):
        run_discrim_model = False
    elif (dmodel is None or doptimizer is None):
        print("Incompatible parameter settings: dmodel = {}   doptimizer = {}".format(dmodel, doptimizer))
        exit(1)

    for epoch in range(n_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0
        running_rec_loss = 0
        running_unweighted_kl_loss = 0
        running_unweighted_ld_loss = 0
        n_total_samples = 0
        kl_loss_weight = kl_loss_schedule(epoch)
        ld_loss_weight = latentdiscrim_loss_schedule(epoch)

        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs, mu, logvar, latentz = model(images)

            # 1. Update discriminator

            if run_discrim_model:
                z_det = latentz.detach()                # stop gradients into encoder
                D_trueprior = dmodel(torch.randn_like(latentz))   # classify true prior samples
                D_latentposterior = dmodel(z_det)                 # classify encoder outputs
            
                dloss = -( torch.log(D_trueprior).mean()
                           + torch.log(1 - D_latentposterior).mean() )
                doptimizer.zero_grad()
                dloss.backward()
                doptimizer.step()

            # 2. Update VAE with 3-part loss (recon, kl, latentdiscrim)

            rec_loss = rec_criterion(outputs, images)
            
            # KL with "free bits" -- basically like a clipped KL
            kl_loss_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss_per_dim_clamped = torch.clamp(kl_loss_per_dim, min=kl_min_loss_per_sample_per_dim)
            kl_loss = kl_loss_per_dim_clamped.sum()

            loss = rec_loss + (kl_loss_weight * kl_loss)

            if run_discrim_model:
                discrim_loss = -torch.log(dmodel(latentz)).sum()
                loss = loss + (ld_loss_weight * discrim_loss)
            else:
                discrim_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

            loss.backward()
            optimizer.step()
            running_rec_loss += rec_loss.item()
            running_unweighted_kl_loss += kl_loss.item()
            running_unweighted_ld_loss += discrim_loss.item()
            running_loss += loss.item()
            n_total_samples += len(images)

        elapsed = time.time() - start_time
        avg_loss = running_loss / n_total_samples
        avg_rec_loss = running_rec_loss / n_total_samples
        avg_kl_loss = running_unweighted_kl_loss / n_total_samples
        avg_ld_loss = running_unweighted_ld_loss / n_total_samples
        print(f"Epoch {epoch+1:2d} | Time: {elapsed:.2f}s | Avg Loss: {avg_loss:.4f} | Avg Rec Loss: {avg_rec_loss:.4f} | Avg Unw KL Loss: {avg_kl_loss:.4f} | KL weight: {kl_loss_weight:.4f} | Avg Unw LD Loss: {avg_ld_loss:.4f} | LD weight: {ld_loss_weight:.4f}")
    
def sample_VAE(model, n_samples):
    samples = model.sample(n_samples).cpu().numpy()
    assert(samples.shape == (n_samples, 1, 28, 28))
    samples = samples.squeeze(1)
    analysis.visualization.display_mnist_inputs_and_outputs(samples, samples, dims=(None,10))
    
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(f"Using device: {device}")

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor()
#    LowPassFilter(cutoff_freq=10)
])

trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)


n_latent_dim = 32

model = models.autoencoders.ConvVAE(1, 28, 28, n_latent_dim)
model.to(device)

Dmodel = models.discriminators.PriorDiscriminator(n_latent_dim)
Dmodel.to(device)



optimizer = optim.Adam(model.parameters())
Doptimizer = optim.Adam(Dmodel.parameters())

n_epochs = 35
n_samples_to_viz = 16

kl_loss_weight_rampup = np.array([0, .1, .1, .2, .2, .3, .3, .4, .5, .6, .7, .8, .9, 1.0])
kl_loss_schedule = lambda epoch: kl_loss_weight_rampup[min(epoch, len(kl_loss_weight_rampup)-1)]
kl_min_loss_per_sample_per_dim = 0.0001

latentdiscrim_loss_weight_rampup = 6.0 * np.array([0, 0, 0, 0, 0, .1, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
latentdiscrim_loss_schedule = lambda epoch: latentdiscrim_loss_weight_rampup[min(epoch, len(latentdiscrim_loss_weight_rampup)-1)]

train_VAE(model,
          trainloader,
          n_epochs,
          optimizer,
          dmodel = Dmodel,
          doptimizer = Doptimizer,
          kl_min_loss_per_sample_per_dim = kl_min_loss_per_sample_per_dim,
          kl_loss_schedule = kl_loss_schedule,
          latentdiscrim_loss_schedule = latentdiscrim_loss_schedule)
evaluate_model(model, testloader)
sample_VAE(model, n_samples_to_viz)
