
from torch import nn
import torch

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding=1, bias=False)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes = 7):
        super(CNN, self).__init__()
        self.MFCC_TOTAL2VECTOR = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.Conv2d(1, 16, 3, padding="same"),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Conv2d(16, 32, 3, padding="same"),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Conv2d(32, 256, 3, padding="same"),
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.LazyLinear(1024)
        )
        self.WAVE_FORM2VECTOR = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.AdaptiveAvgPool1d(4000),
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten(),
            nn.LazyLinear(1024)
        )
        self.MFCC_PARTIAL2VECTOR = nn.Sequential(
            nn.LazyConv2d(64, (3,3), padding="same"),
            nn.ReLU(),
            nn.LazyConv2d(64, (3,3), padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(512)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.LazyLinear(num_classes),
        )

    def loss_function(self, y, target):
        return nn.functional.cross_entropy(y, target)
    
    def forward(self, mfcc_total, **kwargs):
        x = self.MFCC_TOTAL2VECTOR(mfcc_total)
        # y = self.WAVE_FORM2VECTOR(kwargs["wave_form"])
        # x_partial = self.MFCC_PARTIAL2VECTOR(mfcc_partial)
        # x = torch.cat((x_total, x_partial), dim=-1)
        return (self.classifier(torch.concat([x],dim=-1)), )

        
if __name__ == "__main__":
    ae = CNN()
    x = torch.randn(32, 1, 13, 201)
    y = ae(x)
    print(y.shape)