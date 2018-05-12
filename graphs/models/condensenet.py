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

from graphs.weights_initializer import fcn8s_weights_init

class CondenseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def forward(self, x):
        pass
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
    FCN =  CondenseNet(model, 5)
    out = FCN(inp)

    print (out)

if __name__ == '__main__':
    main()

"""
#########################
Model Architecture:
#########################

Input: (N, 3, 32, 32)

"""
