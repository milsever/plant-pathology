"""
Date: 2024-11-04
Description: VAT loss.
Refs:
    https://github.com/9310gaurav/virtual-adversarial-training/blob/master/utils.py
"""

import torch
import torch.nn.functional as F


def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1).mean(dim=0)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


# def _l2_normalize(d):
#     d = d.numpy()
#     d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
#     return torch.from_numpy(d)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_().to(ul_x.device)
    for i in range(num_iters):
        d = xi * _l2_normalize(d)
        d.requires_grad_()
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone()
        model.zero_grad()

    d = _l2_normalize(d)
    r_adv = eps * d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
