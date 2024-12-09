"""
Author: Murat Ilsever
Date: 2023-01-16
Description: Finetuner base class for HeadOnly and HeadThenBody training.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import SGD
from torchsummary import summary

from data import NO_LABEL
from utils import AverageMeterSet


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("Inf")
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class FineTunerBase(object):
    def __init__(self, model, data_loaders, args, result_subdir, device):
        self.model = model
        self.train_dl, self.val_dl = data_loaders
        self.args = args
        self.result_subdir = result_subdir
        self.device = device
        self.class_criterion = None
        self.val_class_criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.best_prec1 = 0
        self.start_epoch = args.start_epoch
        self.early_stopping = None
        self.train_csv = None
        self.val_csv = None

    def finetune(self):
        # Optimizer for HEADONLY training
        self.optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                             lr=self.args.learning_rate,
                             momentum=self.args.momentum,
                             weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.002)

        print("\nFine-tuning strategy: {}.".format(self.args.ftune_strategy.upper()))

        if self.args.ftune_strategy == 'headonly':
            if self.args.resume:
                self._load_checkpoint(self.args.resume)

            self.model.freeze_body()
            print("\nStart HEADONLY training: \n\t{}".format(
                self._get_training_desc().replace(", ", "\n\t")))
            self._pre_headonly_training_hook()
            """Start standart train/validate/logging procedure"""
            self._train_loop()

        elif self.args.ftune_strategy == 'headthenbody':
            """(1) Fine-tune the model head like in 'headonly' training,
               (2) Unfreeze body, and continue fine-tuning whole model.
            """
            if self.args.head_checkpoint:  # if head checkpoint exists, skip to body training
                assert os.path.isfile(self.args.head_checkpoint), \
                    "=> Checkpoint file for HEADONLY training not found at '{}'".format(
                        self.args.head_checkpoint)
                print("Checkpoint file for HEADONLY training found.")
                self._load_checkpoint(self.args.head_checkpoint)
            else:
                self.model.freeze_body()
                print("\nStart HEADONLY training: \n\t{}.".format(
                    self._get_training_desc().replace(", ", "\n\t")))
                self._pre_headonly_training_hook()
                self._train_loop()

            if not self.args.head_checkpoint:
                # Revert to best model state from headonly training,
                self._load_checkpoint(os.path.join(self.result_subdir, "network-head-best.pth"))

            print("Model brought back to its best state from HEADONLY training.")
            print("Best Prec@1 from HEADONLY training: {:.3f}".format(self.best_prec1))

            # Continue with complete model (head and body) training,
            self.start_epoch, self.best_prec1 = 0, 0
            self.model.unfreeze_body()
            summary(self.model, torch.zeros(1, 3, 224, 224).to(self.device), depth=5, verbose=0)

            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=self.args.fine_tuning_lr,
                                 momentum=self.args.momentum,
                                 weight_decay=self.args.weight_decay)

            if self.args.resume:
                self._load_checkpoint(self.args.resume)

            print("\nStart COMPLETE model training: \n\t{}.".format(
                self._get_training_desc().replace(", ", "\n\t")))
            self._pre_complete_model_training_hook()
            self._train_loop()

        elif self.args.ftune_strategy == 'headandbody':
            """Unfreeze body and train the model as a whole."""
            print('Unfreezing body parameters except batch-normalization layers.')
            for module, param in zip(self.model.modules(), self.model.parameters()):
                if not isinstance(module, nn.BatchNorm2d):
                    param.requires_grad = True

            self._train_loop()

        elif self.args.ftune_strategy == 'evaluate':
            assert self.args.resume, "In evaluation mode 'resume' argument must exist."
            self._load_model_state(self.args.resume)
            print("\nEvaluating model.")
            self._validate(0)

    def _train_loop(self):
        self._pre_train_loop_hook()
        if self.args.early_stopping:
            print("Creating early stopping object with patience {}.".format(self.args.patience))
            self.early_stopping = EarlyStopping(patience=self.args.patience)

        for epoch in range(self.start_epoch, self.args.epochs):
            start = time.time()
            if self.model.trainable == "head":
                self._adjust_learning_rate(epoch, self.args.epochs)

            # train for one epoch
            self._train(epoch)

            # evaluate on validation set
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

        self._post_train_loop_hook()

    def _train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def _validate(self, epoch):
        meters = AverageMeterSet()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (img, label) in enumerate(self.val_dl):
            # measure data loading time
            meters.update('Data Time', time.time() - end)

            img = img.to(self.device)
            label = label.to(self.device)

            # compute output
            output = self.model(img)
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

        print(' * Prec@1 {meters[Prec@1].avg:.3f} Prec@5 {meters[Prec@5].avg:.3f}'.format(
            meters=meters))

        if self.val_csv:
            self.val_csv.add_data({'Epoch': epoch, **meters.averages()})

        return meters['Prec@1'].avg, meters['Val Loss'].avg

    def _pre_headonly_training_hook(self):
        pass

    def _pre_complete_model_training_hook(self):
        pass

    def _pre_train_loop_hook(self):
        pass

    def _post_train_loop_hook(self):
        pass

    def _adjust_learning_rate(self, epoch, epoch_f):
        """Learning rate will be decayed by a factor of 0.01 at epoch 5*E/6"""
        lr0 = self.args.learning_rate
        if epoch < int(5 * epoch_f / 6):
            lr = lr0 * (0.01 ** (epoch / int(5 * epoch_f / 6)))
        else:
            lr = lr0 * 0.01

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        # batch_size = target.size(0)
        labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
        return res

    def _load_checkpoint(self, cp_file):
        raise NotImplementedError

    def _load_model_state(self, cp_file):
        assert os.path.isfile(cp_file), "=> Checkpoint file not found at '{}'".format(cp_file)
        print("=> Loading checkpoint '{}' ... ".format(cp_file), end='')
        checkpoint = torch.load(cp_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("done")

    def _save_checkpoint(self, epoch, is_best):
        state = self._get_save_state(epoch)

        if self.args.snapshot_freq != 0 and (epoch % self.args.snapshot_freq) == 0 \
                and epoch != self.start_epoch:
            print('Taking routine snapshot ... ', end='')
            file = os.path.join(self.result_subdir,
                                'network-{}-{:03d}.pth'.format(self.model.trainable, epoch))
            torch.save(state, file)
            print('done.')

        if is_best:
            print('Taking best accuracy snapshot ... ', end='')
            file = os.path.join(self.result_subdir,
                                'network-{}-best.pth'.format(self.model.trainable))
            torch.save(state, file)
            print('done.')

    def _get_save_state(self, epoch):
        raise NotImplementedError

    def _get_training_desc(self):
        raise NotImplementedError


def show_im_pair(indx):
    #     denormalize= T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
    #     im1 = denormalize(img1[indx]).permute(1, 2, 0)
    #     im2 = denormalize(img1_noisy[indx]).permute(1, 2, 0)
    #     plt.figure()
    #     plt.imshow(im1)
    #     plt.figure()
    #     plt.imshow(im2)
    pass
