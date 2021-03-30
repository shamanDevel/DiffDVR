import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .makelayers import _make_layers

# Input: B x C=7 x W x H
#  with B split in half between ground truth and prediction
# with C=7, first three layers: RGB of the prediction/ground truth. 
#           Last four layers: bilinear upscaled input image
class EnhanceNetLargeDiscriminator(nn.Module):
    def __init__(self, resolution, input_channels):
        super(EnhanceNetLargeDiscriminator, self).__init__()
        self.resolution = resolution
        self.input_channels = input_channels
        #input: input_channels x resolution x resolution
        #bring it down to output_channels x 4 x 4
        assert (resolution & (resolution - 1)) == 0, "resolution is not a power of two: %d"%resolution
        config = []
        output_channels = 8
        while resolution > 4:
            output_channels *= 2
            resolution //= 2
            config = config + [output_channels, output_channels, (output_channels, 2)]
        assert(resolution == 4)
        num_linear_features = output_channels * 4 * 4
        # linear layers
        self.features,_ = _make_layers(config, False, in_channels=input_channels)
        self.classifier = nn.Sequential(
            nn.Linear(num_linear_features, 1024, True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1, True),
            #nn.Sigmoid() # no sigmoid at the end, moved to loss function
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

    def forward(self, x):
        b,c,h,w = x.shape
        assert(c==self.input_channels)
        assert(h==self.resolution)
        assert(w==self.resolution)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
