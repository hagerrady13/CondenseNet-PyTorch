"""
Definitions for custom layers and blocks
Adapted from: https://github.com/ShichenLiu/CondenseNet/blob/master/layers.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LearnedGroupConv(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, condense_factor=None, dropout_rate=0.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condense_factor = condense_factor
        self.groups = groups
        self.dropout_rate = dropout_rate

        # Check if given configs are valid
        assert self.in_channels % self.groups == 0, "group value is not divisible by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor is not divisible by input channels"
        assert self.out_channels % self.groups == 0, "group value is not divisible by output channels"

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

        # register conv buffers
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))

        # print(self.in_channels)
        # print(self.out_channels)
        # print(self.groups)
        # print(self.conv.weight.shape)
        # print(self.conv)
        # exit(0)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        ## Dropping here ##
        self.check_if_drop()
        # To mask the output
        weight = self.conv.weight * self.mask
        out_conv = F.conv2d(input=out, weight=weight, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=1)
        return out_conv
    #Sec 3.1: Condensation procedure: number of epochs for each condensing stage: M/2(C-1)
    def check_if_drop(self):
        current_progress = LearnedGroupConv.global_progress
        delta = 0
        # Get current stage
        for i in range(self.condense_factor - 1):   # 3 condensation stages
            if current_progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        # Check for actual dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor # Sec 3.1: Condensation factor: allow each group to select R/C of inputs.
        if delta > 0:
            self.drop(delta)
        return

    def drop(self, delta):
        weight = self.conv.weight * self.mask
        # Sum up all kernels
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    @property
    def lasso_loss(self):
        if self._at_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        # For 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        # Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()


"""
Concerns:
- Mask in convolution??
- how dropping is happening
- During training a fraction of (C−1)/C connections are removed after each of the C-1 condensing stages
- we remove columns in Fg (by zeroing them out) if their L1-norm is small compared to the L1-norm of other columns.
"""

"""
class LearnedGroupConv(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 condense_factor=None, dropout_rate=0.):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._at_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()

"""