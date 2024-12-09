"""
Author: Murat Ilsever
Date: 2024-11-04
Description: Fine-tuner for Virtual Adversarial Training (VAT).
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel

import vat
import report
from utils import AverageMeterSet
from ft_base import FineTunerBase, EarlyStopping


class FineTunerVAT(FineTunerBase):
    def __init__(self, model, data_loaders: tuple, args, result_subdir, device):
        super(FineTunerVAT, self).__init__(model, data_loaders, args, result_subdir, device)
        self.labeled_dl, self.unlabeled_dl = self.train_dl
        self.labeled_dl_iter, self.unlabeled_dl_iter = \
            iter(self.labeled_dl), iter(self.unlabeled_dl)
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

    def _train_loop(self):
        if self.args.early_stopping:
            print("Creating early stopping object with patience {}.".format(self.args.patience))
            self.early_stopping = EarlyStopping(patience=self.args.patience)

        for epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            if self.model.trainable == "head":
                self._adjust_learning_rate(epoch, self.args.epochs)

            # train for one epoch
            self._train(epoch)

            prec1, valid_loss = self._validate(epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)

            print('Epoch {:3d} of {:3d} took {:6.3f}s.'.format(
                epoch + 1, self.args.epochs, time.time() - start))

            self._save_checkpoint(epoch, is_best)

            if self.early_stopping:
                self.early_stopping(valid_loss)

                if self.early_stopping.early_stop:
                    print('Early stopping at epoch {}.'.format(epoch + 1))
                    break

    def _train(self, epoch):
        """ Train for one epoch """
        meters = AverageMeterSet()
        n_iters = self.args.num_iters

        # switch to train mode
        self.model.train()

        end = time.time()
        for it in range(n_iters):
            x, y = next(self.labeled_dl_iter)
            u_x, _ = next(self.unlabeled_dl_iter)

            # measure data loading time
            meters.update('Data Time', time.time() - end)

            x, y, u_x = x.to(self.device), y.to(self.device), u_x.to(self.device)

            # classification loss
            y_pred = self.model(x)
            ce_loss = self.class_criterion(y_pred, y)

            # VAT loss
            u_y = self.model(u_x)
            vat_loss = vat.vat_loss(self.model, u_x, u_y, eps=self.args.epsilon)

            loss = vat_loss + ce_loss

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            meters.update('Sup Loss', ce_loss.item())
            meters.update('VAT Loss', vat_loss.item())

            # measure elapsed time
            meters.update('Batch Time', time.time() - end)
            end = time.time()

            if self.args.print_freq != 0 and it % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}], '
                      'Time: {meters[Batch Time]:.3f}, '
                      'Data: {meters[Data Time]:.3f}, '
                      'Sup: {meters[Sup Loss]:.4f}, '
                      'VAT: {meters[VAT Loss]:.4f}, '.format(
                    epoch + 1, it + 1, n_iters, meters=meters))

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
            'best_prec1': self.best_prec1,
        }

    def _get_training_desc(self):
        desc = "start epoch: {}, epochs: {}, labeled batch size: {}, aux_batch_size: {}, " \
               "learning rate: {}, momentum: {}, weight decay: {}, early stopping: {}, " \
               "patience: {}, num. labeled: {}, num. unlabeled: {}, num. iters: {}, " \
               "epsilon: {}".format(
            self.start_epoch,
            self.args.epochs,
            self.args.labeled_batch_size,
            self.args.aux_batch_size,
            self.args.learning_rate if self.model.trainable == "head" else self.args.fine_tuning_lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.early_stopping,
            self.args.patience if self.args.early_stopping else "N/A",
            len(self.labeled_dl.dataset),
            len(self.unlabeled_dl.dataset),
            self.args.num_iters,
            self.args.epsilon
        )

        return desc
