"""
FCN8s Model with VGG16 encoder
name: fcn8s.py
date: April 2018
"""
import torch
import torch.nn as nn
import cv2
import json
from easydict import EasyDict as edict
import numpy as np

from graphs.weights_initializer import init_model_weights
from graphs.models.denseblock import DenseBlock
from graphs.models.layers import LearnedGroupConv

class CondenseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.stages = self.config.stages
        self.growth_rate = self.config.growth_rate
        assert len(self.stages) == len(self.growth_rate)

        self.init_stride = self.config.init_stride
        self.pool_size = self.config.pool_size
        self.num_classes = self.config.num_classes

        self.progress = 0.0
        self.num_filters = 2 * self.growth_rate[0]
        """
        Initializing layers
        """
        self.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.last_pool = nn.AvgPool2d(self.pool_size)
        self.relu = nn.ReLU(inplace=True)

        self.init_conv = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.num_filters, kernel_size=3, stride=self.init_stride, padding=1, bias=False)

        self.denseblock_one = DenseBlock(num_layers=self.stages[0], in_channels= self.num_filters, growth_rate=self.growth_rate[0], config=self.config)

        self.num_filters += self.stages[0] * self.growth_rate[0]

        self.denseblock_two = DenseBlock(num_layers=self.stages[1], in_channels= self.num_filters, growth_rate=self.growth_rate[1], config=self.config)

        self.num_filters += self.stages[1] * self.growth_rate[1]

        self.denseblock_three = DenseBlock(num_layers=self.stages[2], in_channels= self.num_filters, growth_rate=self.growth_rate[2], config=self.config)

        self.num_filters += self.stages[2] * self.growth_rate[2]
        self.last_bn = nn.BatchNorm2d(self.num_filters)

        self.classifier = nn.Linear(self.num_filters, self.num_classes)

        self.apply(init_model_weights)

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress

        out = self.init_conv(x)

        out = self.denseblock_one(out)
        out = self.transition_pool(out)

        out = self.denseblock_two(out)
        out = self.transition_pool(out)

        out = self.denseblock_three(out)
        out = self.last_bn(out)
        out = self.relu(out)
        out = self.last_pool(out)

        out = out.view(out.size(0), -1)

        out = self.classifier(out)

        return out

"""
Testing model
"""
def load_image(img_path, image_size):
    img = cv2.imread(img_path)
    print ("Original Image shape: " ,img.shape)
    img = cv2.resize(img, (image_size, image_size))
    print ("Reshaped Image: ", img.shape)

    return img

def main():
    config = json.load(open('../../configs/condensenet_exp_0.json'))
    config = edict(config)

    inp = load_image(config.test_image, config.img_size)
    inp = np.swapaxes(inp,0,2)
    inp = np.expand_dims(inp, axis=0)

    inp = torch.autograd.Variable(torch.from_numpy(inp).float())

if __name__ == '__main__':
    main()

"""
#########################
Model Architecture:
#########################

Input: (N, 3, 32, 32)

Conv2D(3, 16, 3, stride=1,padding=1) ->
DenseBlock(num_layers=14, in_channels=16, growth_rate=8)
AvgPool(2,2)
DenseBlock(num_layers=14, in_channels=128, growth_rate=16)
AvgPool(2,2)
DenseBlock(num_layers=14, in_channels=352, growth_rate=32)
BatchNorm(352)
ReLU
AvgPool(8)
Linear(800, 10)
"""
