"""
Author: Murat Ilsever
Date: 2023-01-16
Description: Fine-tuner for Mean-Teacher.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models

import losses
import ramps
import report
from data import NO_LABEL
from utils import AverageMeterSet
from ft_base import FineTunerBase, EarlyStopping
from ft_model import FineTuneModel


class FineTunerMeanTeacher(FineTunerBase):
    def __init__(self, models: tuple, data_loaders: tuple, args, result_subdir, device):
        model, ema_model = models
        super(FineTunerMeanTeacher, self).__init__(model, data_loaders, args, result_subdir, device)
        self.global_step = 0
        self.ema_model = ema_model
        self.class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='sum').to(self.device)
        if args.consistency_type == 'mse':
            self.consistency_criterion = losses.softmax_mse_loss
        elif args.consistency_type == 'kl':
            self.consistency_criterion = losses.softmax_kl_loss

    def _pre_headonly_training_hook(self):
        self.val_student_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_student_headonly.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.val_teacher_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_teacher_headonly.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.train_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'train_headonly.csv'),
                              'Epoch', 'Batch Time', 'Data Time', 'Class Loss', 'EMA Class Loss',
                              'Cons Loss', 'Cons Weight', 'Loss', 'Prec@1', 'Prec@5', 'EMA Prec@1',
                              'EMA Prec@5', 'LR')

    def _pre_complete_model_training_hook(self):
        self.val_student_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_student_complete.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.val_teacher_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_teacher_complete.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.train_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'train_complete.csv'),
                              'Epoch', 'Batch Time', 'Data Time', 'Class Loss', 'EMA Class Loss',
                              'Cons Loss', 'Cons Weight', 'Loss', 'Prec@1', 'Prec@5', 'EMA Prec@1',
                              'EMA Prec@5', 'LR')

        self.global_step = 0
        original_model = models.__dict__[self.args.arch](weights=None)  # No weights - random initialization
        self.ema_model = FineTuneModel(original_model, self.args.arch, self.ema_model.num_classes, self.args.ftune_head,
                                       name="Mean Teacher").to(self.device)
        self.ema_model.freeze_all()  # EMA model is not trainable

    def _post_train_loop_hook(self):
        self.val_student_csv.close()
        self.val_teacher_csv.close()
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

            print("Evaluating the student model:")
            prec1, valid_loss = self._validate(epoch, model=self.model, csv_file=self.val_student_csv)
            print("Evaluating the teacher model:")
            ema_prec1, ema_valid_loss = self._validate(epoch, model=self.ema_model, csv_file=self.val_teacher_csv)

            # remember best prec@1 and save checkpoint
            is_best = ema_prec1 > self.best_prec1
            self.best_prec1 = max(ema_prec1, self.best_prec1)

            print('Epoch {:3d} of {:3d} took {:6.3f}s.'.format(
                epoch + 1, self.args.epochs, time.time() - start))

            self._save_checkpoint(epoch, is_best)

            if self.early_stopping:
                self.early_stopping(ema_valid_loss)

                if self.early_stopping.early_stop:
                    print('Early stopping at epoch {}.'.format(epoch + 1))
                    break

    def _train(self, epoch):
        """ Train for one epoch """
        meters = AverageMeterSet()

        # switch to train mode
        self.model.train()
        self.ema_model.train()

        end = time.time()
        for i, ((noisy1, noisy1_ema), label) in enumerate(self.train_dl):
            # measure data loading time
            meters.update('Data Time', time.time() - end)

            noisy1 = noisy1.to(self.device)
            noisy1_ema = noisy1_ema.to(self.device)
            label = label.to(self.device)
            batch_size = len(label)
            labeled_batch_size = label.data.ne(NO_LABEL).sum()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # compute output
            model_out = self.model(noisy1)
            ema_model_out = self.ema_model(noisy1_ema)

            class_loss = self.class_criterion(model_out, label) / batch_size
            meters.update('Class Loss', class_loss.item())

            ema_class_loss = self.class_criterion(ema_model_out, label) / batch_size
            meters.update('EMA Class Loss', ema_class_loss.item())

            if self.args.consistency:
                consistency_weight = self._get_current_consistency_weight(epoch)
                meters.update('Cons Weight', consistency_weight)
                consistency_loss = consistency_weight * \
                                   self.consistency_criterion(model_out, ema_model_out) / batch_size
                meters.update('Cons Loss', consistency_loss.item())
            else:
                consistency_loss = 0
                meters.update('Cons Loss', 0)

            loss = class_loss + consistency_loss
            assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
            meters.update('Loss', loss.item())

            # measure accuracy and record loss
            prec1, prec5 = self._accuracy(model_out, label, topk=(1, 5))
            meters.update('Prec@1', prec1.item(), labeled_batch_size)
            meters.update('Prec@5', prec5.item(), labeled_batch_size)

            ema_prec1, ema_prec5 = self._accuracy(ema_model_out, label, topk=(1, 5))
            meters.update('EMA Prec@1', ema_prec1.item(), labeled_batch_size)
            meters.update('EMA Prec@5', ema_prec5.item(), labeled_batch_size)

            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            self._update_ema_variables()

            # measure elapsed time
            meters.update('Batch Time', time.time() - end)
            end = time.time()

            if self.args.print_freq != 0 and i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}], '
                      'Time {meters[Batch Time]:.3f}, '
                      'Data {meters[Data Time]:.3f}, '
                      'Class {meters[Class Loss]:.4f}, '
                      'Cons {meters[Cons Loss]:.4f}, '
                      'Prec@1 {meters[Prec@1]:.3f}, '
                      'Prec@5 {meters[Prec@5]:.3f}'.format(
                    epoch + 1, i + 1, len(self.train_dl), meters=meters))

        if self.train_csv:
            self.train_csv.add_data({'Epoch': epoch,
                                     'LR': self.optimizer.param_groups[0]['lr'],
                                     **meters.averages()})

    @torch.no_grad()
    def _validate(self, epoch, model, csv_file):
        meters = AverageMeterSet()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (img, label) in enumerate(self.val_dl):
            # measure data loading time
            meters.update('Data Time', time.time() - end)

            img = img.to(self.device)
            label = label.to(self.device)

            # compute output
            output = model(img)
            loss = self.val_class_criterion(output, label)

            # measure accuracy and record loss
            prec1, prec5 = self._accuracy(output.data, label, topk=(1, 5))
            meters.update('Val Loss', loss.item())
            meters.update('Prec@1', prec1.item())
            meters.update('Prec@5', prec5.item())

            # measure elapsed time
            meters.update('Batch Time', time.time() - end)
            end = time.time()

            if (i % self.args.print_freq) == 0:
                print('Test: [{0}/{1}], '
                      'Time {meters[Batch Time]:.3f}, '
                      'Data {meters[Data Time]:.3f}, '
                      'Class {meters[Val Loss]:.4f}, '
                      'Prec@1 {meters[Prec@1]:.3f}, '
                      'Prec@5 {meters[Prec@5]:.3f}'.format(
                    i + 1, len(self.val_dl), meters=meters))

        print(' * Prec@1 {meters[Prec@1].avg:.3f} Prec@5 {meters[Prec@5].avg:.3f}'.format(meters=meters))

        if csv_file:
            csv_file.add_data({'Epoch': epoch, **meters.averages()})

        return meters['Prec@1'].avg, meters['Val Loss'].avg

    def _update_ema_variables(self):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), self.args.ema_decay)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def _load_checkpoint(self, cp_file):
        assert os.path.isfile(cp_file), "=> Checkpoint file not found at '{}'".format(cp_file)
        print("=> Loading checkpoint '{}' ... ".format(cp_file), end='')
        checkpoint = torch.load(cp_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        if 'ema_state_dict' in checkpoint.keys():
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        self.global_step = checkpoint['global_step'] if 'global_step' in checkpoint.keys() else 0
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        self.best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint.keys() else 0
        print("done")

    def _get_save_state(self, epoch):
        return {
            'epoch': epoch + 1,
            'arch': self.args.arch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'best_prec1': self.best_prec1,
            'optimizer': self.optimizer.state_dict(),
        }

    def _get_training_desc(self):
        desc = "start epoch: {}, epochs: {}, labeled batch size: {}, aux_batch_size: {}, " \
               "learning rate: {}, momentum: {}, weight decay: {}, early stopping: {}, " \
               "patience: {}, num. labeled: {}, num. unlabeled: {}, " \
               "ema decay: {}, consistency type: {}, consistency: {}, " \
               "consistency rampup: {}".format(
            self.start_epoch,
            self.args.epochs,
            self.args.labeled_batch_size,
            self.args.aux_batch_size,
            self.args.learning_rate if self.model.trainable == "head" else self.args.fine_tuning_lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.early_stopping,
            self.args.patience if self.args.early_stopping else "N/A",
            "???",
            "???",
            self.args.ema_decay,
            self.args.consistency_type,
            self.args.consistency,
            self.args.consistency_rampup
        )

        return desc
