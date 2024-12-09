"""
Author: Murat Ilsever
Date: 2024-12-20
Description: Data samplers and transformers.
Refs:
    https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/data.py
"""

import itertools

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

NO_LABEL = -1


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def split_idxs(dataset, num_labels):
    labeled_idxs, unlabeled_idxs = [], []
    # randomize label indicies
    indxs = torch.randperm(len(dataset)).numpy()

    num_classes = len(dataset.classes)

    if num_labels < 1:
        max_count = np.asarray(num_labels * np.bincount(dataset.targets), dtype=int)
        print("Keeping {:.1f}% labeles per class: {}.".format(num_labels * 100, max_count))
    else:
        max_count = np.asarray([num_labels] * num_classes, dtype=int)
        print("Keeping {} labels per class.".format(max_count[0]))

    count = [0] * num_classes
    for i in indxs:
        path, label = dataset.imgs[i]
        if count[label] < max_count[label]:
            labeled_idxs.append(i)
        else:
            unlabeled_idxs.append(i)
            dataset.targets[i] = NO_LABEL
            dataset.imgs[i] = path, NO_LABEL
        count[label] += 1

    return labeled_idxs, unlabeled_idxs


class InfiniteSampler(Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
