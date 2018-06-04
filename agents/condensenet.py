import numpy as np

from tqdm import tqdm
import shutil

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from graphs.models.condensenet import CondenseNet
from graphs.losses.loss import CrossEntropyLoss2d
from datasets.cifar10 import Cifar10DataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate, cls_accuracy
from utils.misc import print_cuda_statistics
from utils.train_utils import adjust_learning_rate

cudnn.benchmark = True


class CondenseNetAgent:
    """
    This class will be responsible for handling the whole process of our architecture.
    """

    def __init__(self, config):
        self.config = config
        # Create an instance from the Model
        self.model = CondenseNet(self.config)
        #print(self.model)
        # Create an instance from the data loader
        self.data_loader = Cifar10DataLoader(self.config)
        # Create instance from the loss
        self.loss = CrossEntropyLoss2d()
        # Create instance from the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.config.learning_rate,
                                         momentum=float(self.config.momentum),
                                         weight_decay=self.config.weight_decay,
                                         nesterov=True)
        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        # Set the seed of torch
        torch.manual_seed(self.config.seed)

        if self.cuda:
            print("Operation will be on *****GPU-CUDA***** ")
            torch.cuda.manual_seed_all(self.config.seed)
            print_cuda_statistics()

            # Get my models to run on CUD
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
        else:
            print("Operation will be on *****CPU***** ")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='CondenseNet')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))

        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters
        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)

            # current iteration over total iterations
            progress = float(self.current_epoch * self.data_loader.train_iterations + current_batch) / (self.config.max_epoch * self.data_loader.train_iterations)
            #progress = float(self.current_iteration) / (self.config.max_epoch * self.data_loader.train_iterations)
            x, y = Variable(x), Variable(y)
            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch, nBatch= self.data_loader.train_iterations)
            # model
            pred = self.model(x, progress)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.cpu().data[0])):
                raise ValueError('Loss is nan during training...')
            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1,5))

            epoch_loss.update(cur_loss.data[0])
            if self.cuda:
                top1_acc.update(top1.cpu().numpy()[0], x.size(0))
                top5_acc.update(top5.cpu().numpy()[0], x.size(0))
            else:
                top1_acc.update(top1.numpy()[0], x.size(0))
                top5_acc.update(top5.numpy()[0], x.size(0))

            self.current_iteration += 1
            current_batch += 1

            self.summary_writer.add_scalar("epoch/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/accuracy", top1_acc.val, self.current_iteration)
        tqdm_batch.close()

        print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)

            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.data[0])):
                raise ValueError('Loss is nan during training...')

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1,5))
            epoch_loss.update(cur_loss.data[0])
            if self.cuda:
                top1_acc.update(top1.cpu().numpy()[0], x.size(0))
                top5_acc.update(top5.cpu().numpy()[0], x.size(0))
            else:
                top1_acc.update(top1.numpy()[0], x.size(0))
                top5_acc.update(top5.numpy()[0], x.size(0))


        print("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.avg) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
