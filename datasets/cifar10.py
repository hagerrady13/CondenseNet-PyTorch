import numpy as np

import torch
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
import torchvision.datasets as v_datasets

from torch.utils.data import DataLoader, TensorDataset, Dataset


class Cifar10DataLoader:
    def __init__(self, config):
        self.config = config

        if config.data_mode == "numpy_train":

            print("Loading DATA.....")
            train_data = torch.from_numpy(np.load(config.data_folder + config.x_train)).float()
            train_labels = torch.from_numpy(np.load(config.data_folder + config.y_train)).long()
            valid_data = torch.from_numpy(np.load(config.data_folder + config.x_valid)).float()
            valid_labels = torch.from_numpy(np.load(config.data_folder + config.y_valid)).long()

            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            print("""
Some Statistics about the training data
train_data shape: {}, type: {}
train_labels shape: {}, type: {}
valid_data shape: {}, type: {}
valid_labels type: {}, type: {}
train_iterations: {}
valid_iterations: {}
            """.format(train_data.size(), train_data.type(), train_labels.size(), train_labels.type(),
                       valid_data.size(), valid_data.type(), valid_labels.size(), valid_labels.type(),
                       self.train_iterations, self.valid_iterations))

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        elif config.data_mode == "numpy_test":
            test_data = torch.from_numpy(np.load(config.data_folder + config.x_test)).float()
            test_labels = torch.from_numpy(np.load(config.data_folder + config.y_test)).long()

            self.len_test_data = test_data.size()[0]

            self.test_iterations = (self.len_test_data + self.config.batch_size - 1) // self.config.batch_size

            print("""
Some Statistics about the testing data
test_data shape: {}, type: {}
test_labels shape: {}, type: {}
test_iterations: {}
            """.format(test_data.size(), test_data.type(), test_labels.size(), test_labels.type(),
                       self.test_iterations))

            test = TensorDataset(test_data, test_labels)

            self.test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)

        elif config.data_mode == "random":
            train_data = torch.randn(2, self.config.num_channels, self.config.img_size, self.config.img_size)
            train_labels = torch.ones(2, self.config.img_size, self.config.img_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
