import torch
from torch import nn


class MLPImageAutoencoder(nn.Module):
    def __init__(self, img_shape, hidden_dim):
        super().__init__()
        assert(len(img_shape)==2)

        D = img_shape[0] * img_shape[1]

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, D),
            nn.Unflatten(1, img_shape)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class ConvSameBlock(nn.Module):
    def __init__(self, in_ch, out_ch, nonlinearity=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nonlinearity(),
        )
    def forward(self, x):
        return self.net(x)

    
class ConvDownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        return self.net(x)

class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, nonlinearity=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nonlinearity(),
        )
    def forward(self, x):
        return self.net(x)

    
class ConvAE(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvSameBlock(in_ch, 16),
            ConvSameBlock(16, 16),
            ConvDownsampleBlock(16, 16),
            ConvDownsampleBlock(16, 32),
        )

        self.decoder = nn.Sequential(
            ConvUpsampleBlock(32, 16),
            ConvUpsampleBlock(16, 16),
            ConvSameBlock(16, 16),
            ConvSameBlock(16, in_ch, nonlinearity=nn.Sigmoid)
        )

    def forward(self, x):
        B,C,H,W = x.shape
        assert H % 4 == 0 and W % 4 == 0, "H and W must be divisible by 4"
        z = self.encoder(x)
        return self.decoder(z)
    

class ConvVAE(nn.Module):
    def __init__(self, in_ch, H, W, code_dim):
        super().__init__()

        ch2 = 16
        ch3 = 32

        self.flattened_encoder = nn.Sequential(
            ConvSameBlock(in_ch, ch2),
            ConvDownsampleBlock(ch2, ch2),
            ConvDownsampleBlock(ch2, ch3),
            nn.Flatten()
        )

        dH, dW = H//4, W//4

        self.feat_dim = dH * dW * ch3
        self.code_dim = code_dim

        self.fc_mu = nn.Linear(self.feat_dim, self.code_dim)
        self.fc_logvar = nn.Linear(self.feat_dim, self.code_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.code_dim, self.feat_dim),
            nn.Unflatten(in_ch, (ch3, dH, dW)),
            ConvUpsampleBlock(ch3, ch2),
            ConvUpsampleBlock(ch2, ch2),
            ConvSameBlock(ch2, in_ch, nonlinearity=nn.Sigmoid)
        )

    def sample(self, n_samples):
        device = next(self.parameters()).device
        eps    = torch.randn(n_samples, self.code_dim, device=device)
        self.eval()
        with torch.no_grad():
            return self.decoder(eps)

    def forward(self, x):
        B,C,H,W = x.shape
        assert H % 4 == 0 and W % 4 == 0, "H and W must be divisible by 4"
        feat = self.flattened_encoder(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z   = mu + eps*std

        recon = self.decoder(z)

        return recon, mu, logvar, z
    
