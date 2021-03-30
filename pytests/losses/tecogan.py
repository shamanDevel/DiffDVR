import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Input: B x C=7 x W x H
#  with B split in half between ground truth and prediction
# with C=7, first three layers: RGB of the prediction/ground truth. 
#           Last four layers: bilinear upscaled input image
class TecoGANDiscriminator(nn.Module):
    def __init__(self, resolution, input_channels):
        super(TecoGANDiscriminator, self).__init__()
        self.resolution = resolution
        self.input_channels = input_channels
        #input: input_channels x resolution x resolution
        #bring it down to output_channels x 32 x 32
        assert (resolution & (resolution - 1)) == 0, "resolution is not a power of two: %d"%resolution
        modules = []
        channels = input_channels
        while resolution > 32:
            resolution //= 2
            modules = modules + [
                nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                ]
            channels = 64
        # output size: 64x32x32
        # and then add remaining features
        modules = modules + [
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), #128x16x16
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), #256x8x8
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False), #256x4x4
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
            ]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1, True),
            #nn.Sigmoid(), no sigmoid at the end, moved to loss function
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        b,c,h,w = x.shape
        assert(c==self.input_channels)
        assert(h==self.resolution)
        assert(w==self.resolution)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
