
from torch import nn
import torch

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding=1, bias=False)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                    stride=stride, padding=2, bias=False)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(out_channels, out_channels)
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


from .TIMNET import TIMNET, WeightLayer
class CNN(nn.Module):
    def __init__(self, num_classes = 7):
        super(CNN, self).__init__()
        # self.MFCC_TOTAL2VECTOR = nn.Sequential(
        #     nn.LazyConv2d(64, (5,5), padding=2),
        #     nn.ReLU(),
        #     nn.LazyBatchNorm2d(),
        #     ResidualBlock(64, 64),
        #     ResidualBlock(64, 64),
        #     nn.AdaptiveAvgPool2d((8, 8)),
        #     nn.Flatten(),
        #     nn.LazyLinear(128)
        # )
        # self.MFCC_PARTIAL2VECTOR = nn.Sequential(
        #     nn.LazyConv2d(64, (5,5), padding=2),
        #     nn.ReLU(),
        #     nn.LazyBatchNorm2d(),
        #     ResidualBlock(64, 64),
        #     ResidualBlock(64, 64),
        #     nn.AdaptiveAvgPool2d((8, 8)),
        #     nn.Flatten(start_dim=-3),
        #     nn.LazyLinear(128)
        # )
        # self.RNN = nn.LSTM(128, 64, 2, batch_first=True, bidirectional=True)
        # self.Attention = nn.MultiheadAttention(128, 8, batch_first=True)

        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.LazyLinear(num_classes),
        # )
        self.TIMNET = TIMNET(nb_filters=39, kernel_size=2, nb_stacks=1, dilations=None, activation="relu", dropout_rate=0.1, return_sequences=True)
        self.WEIGHTLAYER = WeightLayer()
        self.classifier = nn.Sequential(
            nn.LazyLinear(num_classes),
        )

    def loss_function(self, y, target):
        return nn.functional.cross_entropy(y, target)
    
    def forward(self, mfcc_total, mfcc_partial=None):
        # x = self.MFCC_TOTAL2VECTOR(mfcc_total).unsqueeze(1) # b, seq_len, feature
        # mfcc_partial b,seq_len,channel,width,height
        # b,seq_len,channel,widht,height = mfcc_partial.size()
        # mfcc_partial = mfcc_partial.view(b*seq_len, channel, widht, height)
        # x_partial = self.MFCC_PARTIAL2VECTOR(mfcc_partial)
        # x_partial = x_partial.view(b, seq_len, -1)
        # x_partial, _ = self.RNN(x_partial)
        # x, _ = self.Attention(x, x_partial, x_partial)
        # x = x.squeeze(1)
        x = self.TIMNET(mfcc_total)
        x = self.WEIGHTLAYER(x)

        return self.classifier(x) 

        
if __name__ == "__main__":
    ae = CNN()
    x = torch.randn(32, 1, 13, 201)
    y = ae(x)
    print(y.shape)