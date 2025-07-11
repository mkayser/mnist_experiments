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
    
