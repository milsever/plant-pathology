"""
Author: Murat Ilsever
Date: 2024-11-04
Description: Fine-tuner for MixMatch.
"""

import os
import time

import torch
import torch.nn.parallel
import torchvision.models as models

import mixmatch
import report
from utils import AverageMeterSet
from ft_base import FineTunerBase, EarlyStopping
from ft_model import FineTuneModel


class FineTunerMixMatch(FineTunerBase):
    def __init__(self, models: tuple, data_loaders: tuple, args, result_subdir, device):
        model, ema_model = models
        super(FineTunerMixMatch, self).__init__(model, data_loaders, args, result_subdir, device)
        self.labeled_dl, self.unlabeled_dl = self.train_dl
        self.labeled_dl_iter, self.unlabeled_dl_iter = \
            iter(self.labeled_dl), iter(self.unlabeled_dl)
        self.global_step = 0
        self.ema_model = ema_model
        self.MixMatch = mixmatch.MixMatchLoss(args, self.model.num_classes, self.device)
        self.ema_optimizer = mixmatch.WeightEMA(self.model, self.ema_model, 0, args.ema_decay)

    def _pre_headonly_training_hook(self):
        self.val_model_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_model_headonly.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.val_ema_model_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_ema_model_headonly.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.train_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'train_headonly.csv'),
                              'Epoch', 'Batch Time', 'Data Time', 'Sup. Loss', 'Unsup. Loss',
                              'Lam_U', 'LR')

    def _pre_complete_model_training_hook(self):
        self.val_model_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_model_complete.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.val_ema_model_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'validate_ema_model_complete.csv'),
                              'Epoch', 'Batch Time', 'Val Loss', 'Prec@1', 'Prec@5')
        self.train_csv = \
            report.GenericCSV(os.path.join(self.result_subdir, 'train_headonly.csv'),
                              'Epoch', 'Batch Time', 'Data Time', 'Sup. Loss', 'Unsup. Loss',
                              'Lam_U', 'LR')

        self.global_step = 0
        original_model = models.__dict__[self.args.arch](weights=None)
        self.ema_model = FineTuneModel(original_model, self.args.arch, self.ema_model.num_classes,
                                       self.args.ftune_head,
                                       name="MixMatch").to(self.device)
        self.ema_model.freeze_all()  # EMA model is not trainable
        self.ema_optimizer = mixmatch.WeightEMA(self.model, self.ema_model, 0, self.args.ema_decay)

    def _post_train_loop_hook(self):
        self.val_model_csv.close()
        self.val_ema_model_csv.close()
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

            print("Evaluating the current model:")
            prec1, valid_loss = self._validate(epoch, model=self.model,
                                               csv_file=self.val_model_csv)
            print("Evaluating the EMA model:")
            ema_prec1, ema_valid_loss = self._validate(epoch, model=self.ema_model,
                                                       csv_file=self.val_ema_model_csv)

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
        n_iters = self.args.num_iters

        # switch to train mode
        self.model.train()

        end = time.time()
        for it in range(n_iters):
            x, y = next(self.labeled_dl_iter)
            u_x, _ = next(self.unlabeled_dl_iter)

            # measure data loading time
            meters.update('Data Time', time.time() - end)

            # forward inputs
            current = epoch + it / n_iters
            input = {'model': self.model,
                     'u_x': u_x,
                     'x': x,
                     'y': y,
                     'current': current}

            # compute mixmatch loss
            loss_x, loss_u, lam_u = self.MixMatch(input)
            loss = loss_x + lam_u * loss_u

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_optimizer.step()
            self.global_step += 1

            meters.update('Sup. Loss', loss_x.item())
            meters.update('Unsup. Loss', lam_u * loss_u.item())

            # measure elapsed time
            meters.update('Batch Time', time.time() - end)
            end = time.time()

            if self.args.print_freq != 0 and it % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}], '
                      'Time: {meters[Batch Time]:.3f}, '
                      'Data: {meters[Data Time]:.3f}, '
                      'Sup: {meters[Sup. Loss]:.4f}, '
                      'Unsup: {meters[Unsup. Loss]:.4f}, '
                      'Lam_U: {3:.3f}, '.format(
                    epoch + 1, it + 1, n_iters, lam_u, meters=meters))

        if self.train_csv:
            self.train_csv.add_data({'Epoch': epoch,
                                     'LR': self.optimizer.param_groups[0]['lr'],
                                     'Lam_U': lam_u,
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
            batch_size = len(label)

            # compute output
            output = model(img)
            loss = self.val_class_criterion(output, label)

            # measure accuracy and record loss
            prec1, prec5 = self._accuracy(output.data, label, topk=(1, 5))
            meters.update('Val Loss', loss.item(), batch_size)
            meters.update('Prec@1', prec1.item(), batch_size)
            meters.update('Prec@5', prec5.item(), batch_size)

            # measure elapsed time
            meters.update('Batch Time', time.time() - end)
            end = time.time()

            if (i % self.args.print_freq) == 0:
                print('Test: [{0}/{1}], '
                      'Time {meters[Batch Time]:.3f}, '
                      'Data {meters[Data Time]:.3f}, '
                      'Loss {meters[Val Loss]:.4f}, '
                      'Prec@1 {meters[Prec@1]:.3f}, '
                      'Prec@5 {meters[Prec@5]:.3f}'.format(
                    i + 1, len(self.val_dl), meters=meters))

        print(' * Prec@1 {meters[Prec@1].avg:.3f} Prec@5 {meters[Prec@5].avg:.3f}'.format(
            meters=meters))

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
               "patience: {}, num. labeled: {}, num. unlabeled: {}, num. iters: {}, " \
               "ema decay: {}, K: {}, T: {}, lambda U: {}, alpha: {}".format(
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
            self.args.ema_decay,
            self.args.K,
            self.args.T,
            self.args.lambda_u,
            self.args.alpha
        )

        return desc
