import torch
from torchvision import datasets, transforms
from torch import nn, optim
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import models.autoencoders


def display_mnist_inputs_and_outputs(input_imgs, output_imgs, input_fft_imgs, output_fft_imgs, dims=(None,8)):
    def ceil_int_div(a,b):
        return (a+b-1) // b

    def place_image(ax, img, caption, cmap):
        ax.set_aspect('equal')
        ax.set_title(caption)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    
    n_quads = len(input_imgs)
    n_images = n_quads * 4
    image_group_size = 4
    
    # establish grid dimensions
    assert(len(dims)==2)
    n_rows, n_cols = dims
    assert(not(n_rows is None and n_cols is None))
    if(n_rows is None):
        n_rows = ceil_int_div(n_images, n_cols)
    if(n_cols is None):
        n_cols = ceil_int_div(n_images, n_rows)
        if(n_cols % image_group_size != 0):
            n_cols += (image_group_size - (n_cols % image_group_size))
    assert n_rows * n_cols >= n_images, "NRows={},  NCols={},   NImages={}".format(n_rows, n_cols, n_images)
    assert(n_cols % 2 == 0)

    # Build figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8, 8),  # total inches
                             dpi=100,         # dots per inch
                             constrained_layout=True)

    # Populate each pair of cells
    for i in range(n_quads):
        input_img = input_imgs[i]
        output_img = output_imgs[i]
        input_fft_image = input_fft_imgs[i]
        output_fft_image = output_fft_imgs[i]
        
        j = i * image_group_size
        r = j // n_cols
        c = j % n_cols
        assert(c+1 < n_cols)
        assert(c+2 < n_cols)
        assert(c+3 < n_cols)
        place_image(axes[r][c], input_img, "Input", "gray")
        place_image(axes[r][c+1], output_img, "Output", "gray")
        place_image(axes[r][c+2], input_fft_image, "Input_FFT", "magma")
        place_image(axes[r][c+3], output_fft_image, "Output_FFT", "magma")

    plt.show()

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


# Simple MLP model
model = models.autoencoders.ConvAE(1)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
n_epochs = 10

# Training loop
for epoch in range(n_epochs):
    model.train()
    start_time = time.time()
    running_loss = 0

    for images, _ in trainloader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    elapsed = time.time() - start_time
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1:2d} | Time: {elapsed:.2f}s | Avg Loss: {avg_loss:.4f}")

# Evaluation loop
model.eval()
total_loss = 0
total = 0

all_cpu_np_images = []
all_cpu_np_outputs = []

with torch.no_grad():
    for images, _ in testloader:
        images = images.to(device)
        outputs = model(images)
        losses = criterion(outputs, images)
        total += images.size(0)
        total_loss += losses.sum()

        cpu_np_images = images.cpu().numpy()
        cpu_np_outputs = outputs.detach().cpu().numpy()

        all_cpu_np_images.append(cpu_np_images)
        all_cpu_np_outputs.append(cpu_np_outputs)


average_loss = total_loss / total
print(f"\nAverage Loss: {average_loss:.3f}%")


images_tensor = np.concatenate(all_cpu_np_images, axis=0)
outputs_tensor = np.concatenate(all_cpu_np_outputs, axis=0)

N = images_tensor.shape[0]
idx = np.random.permutation(N)[:24]
images_tensor = images_tensor[idx]
outputs_tensor = outputs_tensor[idx]

# For the purpose of image display, we eliminate the channels axis
images_tensor = images_tensor.squeeze(1)  # now (N,28,28)
outputs_tensor = outputs_tensor.squeeze(1)  # now (N,28,28)


def compute_fft_images(batch):
    # compute 2D FFT over each image in one shot:
    F = np.fft.fft2(batch, axes=(1,2))     # shape (N,28,28), complex
    Fshift = np.fft.fftshift(F, axes=(1,2))
    magnitude = np.log1p(np.abs(Fshift))   # shape (N,28,28) real
    # now magnitude[i] is the FFT‐magnitude image for batch[i]
    return magnitude

images_fft = compute_fft_images(images_tensor)
outputs_fft = compute_fft_images(outputs_tensor)

display_mnist_inputs_and_outputs(images_tensor, outputs_tensor, images_fft, outputs_fft)
