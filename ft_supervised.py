"""
Author: Murat Ilsever
Date: 2023-01-16
Description: Fine-tuner for Supervised training.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel

import report
from data import NO_LABEL
from utils import AverageMeterSet
from ft_base import FineTunerBase


class FineTunerSupervised(FineTunerBase):
    def __init__(self, model, data_loaders, args, result_subdir, device):
        super(FineTunerSupervised, self).__init__(model, data_loaders, args, result_subdir, device)
        self.class_criterion = nn.CrossEntropyLoss().to(self.device)

    def _pre_headonly_training_hook(self):
        self.val_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_headonly.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.train_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'train_headonly.csv'.format(self.model.trainable)),
                              'Epoch', 'Batch Time', 'Data Time', 'Train Loss', 'Prec@1', 'Prec@5', 'LR')

    def _pre_complete_model_training_hook(self):
        self.val_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_complete.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.train_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'train_complete.csv'.format(self.model.trainable)),
                              'Epoch', 'Batch Time', 'Data Time', 'Train Loss', 'Prec@1', 'Prec@5', 'LR')

    def _post_train_loop_hook(self):
        self.val_csv.close()
        self.train_csv.close()

    def _train(self, epoch):
        meters = AverageMeterSet()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (img, label) in enumerate(self.train_dl):
            assert (label == NO_LABEL).sum() == 0
            # measure data loading time
            meters.update('Data Time', time.time() - end)

            img = img.to(self.device)
            label = label.to(self.device)
            batch_size = len(label)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # compute output
            output = self.model(img)
            loss = self.class_criterion(output, label)

            # measure accuracy and record loss
            prec1, prec5 = self._accuracy(output.data, label, topk=(1, 5))
            meters.update('Train Loss', loss.item(), batch_size)
            meters.update('Prec@1', prec1.item(), batch_size)
            meters.update('Prec@5', prec5.item(), batch_size)

            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            meters.update('Batch Time', time.time() - end)
            end = time.time()

            if self.args.print_freq != 0 and i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}], '
                      'Time {meters[Batch Time]:.3f}, '
                      'Data {meters[Data Time]:.3f}, '
                      'Class {meters[Train Loss]:.4f}, '
                      'Prec@1 {meters[Prec@1]:.3f}, '
                      'Prec@5 {meters[Prec@5]:.3f}'.format(
                    epoch + 1, i + 1, len(self.train_dl), meters=meters))

        if self.train_csv:
            self.train_csv.add_data({'Epoch': epoch,
                                     'LR': self.optimizer.param_groups[0]['lr'],
                                     **meters.averages()})

    def _load_checkpoint(self, cp_file):
        assert os.path.isfile(cp_file), "=> Checkpoint file not found at '{}'".format(cp_file)
        print("=> Loading checkpoint '{}' ... ".format(cp_file), end='')
        checkpoint = torch.load(cp_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        self.best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint.keys() else 0
        print("done")

    def _get_save_state(self, epoch):
        return {
            'epoch': epoch + 1,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_prec1': self.best_prec1
        }

    def _get_training_desc(self):
        desc = "start epoch: {}, epochs: {}, batch size: {}, " \
               "learning rate: {}, momentum: {}, weight decay: {}, early stopping: {}, " \
               "patience: {}, all labels: {}, exclude unlabeled: {}, num. labeled: {}".format(
            self.start_epoch,
            self.args.epochs,
            self.args.labeled_batch_size,
            self.args.learning_rate if self.model.trainable == "head" else self.args.fine_tuning_lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.early_stopping,
            self.args.patience if self.args.early_stopping else "N/A",
            self.args.all_labels,
            self.args.exclude_unlabeled,
            len(self.train_dl.sampler)
        )

        return desc
