from torch import nn
import torch

class MFCC_AE(nn.Module):
    def __init__(self, num_classes = 7):
        super(MFCC_AE, self).__init__()
        self.MFCC2VECTOR = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(512)
        )
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
        )

    def loss_function(self, recon_x, x, y, target):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        ce = nn.functional.cross_entropy(y, target)
        return 1e-3 * recon_loss + ce
    
    def forward(self, x):
        x = self.MFCC2VECTOR(x) # [32, 400]
        ae_x = self.encoder(x)
        recon_x = self.decoder(ae_x)
        return recon_x,x, self.classifier(ae_x)

        
if __name__ == "__main__":
    ae = MFCC_AE()
    x = torch.randn(32, 1, 13, 201)
    recon_x, x, y = ae(x)
    print(recon_x.shape, x.shape, y.shape)