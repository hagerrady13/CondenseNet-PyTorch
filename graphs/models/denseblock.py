"""
Definitions for custom blocks
"""
import torch
import torch.nn as nn
from graphs.models.layers import LearnedGroupConv

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, config):
        super().__init__()

        for layer_id in range(num_layers):
            layer = DenseLayer(in_channels=in_channels + (layer_id * growth_rate), growth_rate=growth_rate, config=config)
            self.add_module('dense_layer_%d' % (layer_id + 1), layer)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, config):
        super().__init__()
        self.config = config
        self.conv_bottleneck = self.config.conv_bottleneck
        self.group1x1 = self.config.group1x1
        self.group3x3 = self.config.group3x3
        self.condense_factor = self.config.condense_factor
        self.dropout_rate = self.config.dropout_rate

        # 1x1 conv in_channels --> bottleneck*growth_rate
        self.conv_1 = LearnedGroupConv(in_channels=in_channels, out_channels=self.conv_bottleneck * growth_rate, kernel_size=1,
                                       groups=self.group1x1, condense_factor=self.condense_factor, dropout_rate=self.dropout_rate)

        self.batch_norm = nn.BatchNorm2d(self.conv_bottleneck * growth_rate)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 conv bottleneck*growth_rate --> growth_rate
        self.conv_2 = nn.Conv2d(in_channels=self.conv_bottleneck * growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, stride=1, groups=self.group3x3, bias=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv_2(out)

        return torch.cat([x, out], 1)

"""
---------------------------------
(denseblock_one): DenseBlock(
(dense_layer_1): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_2): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(24, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_3): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_4): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(40, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_5): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_6): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(56, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_7): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_8): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(72, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_9): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(80, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_10): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(88, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_11): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_12): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(104, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_13): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(112, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_14): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
)
---------------------------------
(denseblock_two): DenseBlock(
(dense_layer_1): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_2): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(144, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_3): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_4): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(176, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_5): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_6): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(208, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_7): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(224, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_8): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_9): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_10): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(272, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_11): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_12): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(304, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(304, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_13): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_14): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(336, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
)
---------------------------------
(denseblock_three): DenseBlock(
(dense_layer_1): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_2): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_3): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_4): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_5): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_6): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_7): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_8): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_9): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_10): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_11): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_12): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_13): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
(dense_layer_14): DenseLayer(
  (conv_1): LearnedGroupConv(
    (batch_norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True)
    (relu): ReLU(inplace)
    (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
  (relu): ReLU(inplace)
  (conv_2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
)
)
---------------------------------
"""