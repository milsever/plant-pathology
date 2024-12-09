"""
Date: 2024-11-04
Description: MixMatch loss.
Refs:
    https://github.com/Jeffkang-94/Mixmatch-pytorch-SSL/blob/master/SSL_loss/mixmatch.py
    https://github.com/Jeffkang-94/Mixmatch-pytorch-SSL/blob/master/model/ema.py
    https://github.com/Jeffkang-94/Mixmatch-pytorch-SSL/blob/master/data_loader/loader.py
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Augmentation:
    def __init__(self, K, transform):
        self.transform = transform
        self.K = K

    def __call__(self, x):
        # Applying stochastic augmentation K times
        out = [self.transform(x) for _ in range(self.K)]
        return out


class MixMatchLoss(nn.Module):
    def __init__(self, args, num_classes, device):
        super().__init__()
        self.K = args.K
        self.T = args.T
        self.bt = args.labeled_batch_size
        self.epochs = args.epochs
        self.lambda_u = args.lambda_u
        self.beta_dist = torch.distributions.beta.Beta(args.alpha, args.alpha)
        self.num_classes = num_classes
        self.device = device

    def sharpen(self, y):
        y = y.pow(1 / self.T)
        return y / y.sum(dim=1, keepdim=True)

    def cal_loss(self, logit_x, y, logit_u_x, y_hat, epoch, lambda_u):
        """
        :param logit_x   : f(x)
        :param y         : true target of x
        :param logit_u_x : f(u_x)
        :param y_hat     : guessed label of u_x
        :param epoch     : current epoch
        :param lambda_u  : linearly increase the weight from 0 to lambda_u
        :return          : CE loss of x, mse loss of (f(u_x), y_hat), weight of u_x
        """
        probs_u = torch.softmax(logit_u_x, dim=1)  # score of u_x
        loss_x = -torch.mean(torch.sum(F.log_softmax(logit_x, dim=1) * y, dim=1))  # Cross entropy
        loss_u_x = F.mse_loss(probs_u, y_hat)  # MSE loss
        linear_weight = float(np.clip(epoch / self.epochs, 0.0,
                                      1.0))  # linearly ramp up the contribution of unlabeled set

        return loss_x, loss_u_x, lambda_u * linear_weight

    def mixup(self, all_inputs, all_targets):
        lam = self.beta_dist.sample().item()
        lam = max(lam, 1 - lam)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = lam * input_a + (1 - lam) * input_b
        mixed_target = lam * target_a + (1 - lam) * target_b
        mixed_input = list(torch.split(mixed_input, self.bt))
        mixed_input = mixmatch_interleave(mixed_input, self.bt)
        return mixed_input, mixed_target

    def forward(self, input):
        x = input['x']
        y = input['y']
        u_x = [x for x in input['u_x']]
        current = input['current']
        model = input['model']

        # make onehot label
        y = F.one_hot(y, self.num_classes)
        x, y = x.to(self.device), y.to(self.device)
        u_x = [i.to(self.device) for i in u_x]

        with torch.no_grad():
            y_hat = sum([model(k).softmax(dim=1) for k in u_x]) / self.K
            y_hat = self.sharpen(y_hat)
            y_hat.detach_()

        # mixup
        all_inputs = torch.cat([x] + u_x, dim=0)
        all_targets = torch.cat([y] + [y_hat] * self.K, dim=0)
        mixed_input, mixed_target = self.mixup(all_inputs, all_targets)

        logit = [model(mixed_input[i]) for i in range(len(mixed_input))]
        logits = mixmatch_interleave(logit, self.bt)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        loss_x, loss_u, lam_u = self.cal_loss(logits_x, mixed_target[:self.bt], logits_u,
                                              mixed_target[self.bt:], current, self.lambda_u)
        return loss_x, loss_u, lam_u


def mixmatch_interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def mixmatch_interleave(xy, batch):
    nu = len(xy) - 1
    offsets = mixmatch_interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


class WeightEMA(object):
    def __init__(self, model, ema_model, wd, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = wd

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)
