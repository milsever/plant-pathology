"""
Author: Murat Ilsever
Date: 2022-12-13
Description: Starts fine-tuning a given model based on a given fine-tuning strategy.
"""

import os
import sys
from collections.abc import Sequence

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchsummary import summary
import numpy as np

from ft_model import FineTuneModel
from ft_supervised import FineTunerSupervised
from ft_meanteacher import FineTunerMeanTeacher
from ft_mixmatch import FineTunerMixMatch
from ft_vat import FineTunerVAT
from data import TwoStreamBatchSampler, InfiniteSampler, TransformTwice, split_idxs
import mixmatch

import report
from utils import *
from cli import parse_commandline_args
from image_utils import save_image

args = None
result_subdir = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_descriptor = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights.DEFAULT",
    "densenet121": "DenseNet121_Weights.DEFAULT",
    "densenet201": "DenseNet201_Weights.DEFAULT",
    "inception_v3": "Inception_V3_Weights.DEFAULT",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights.DEFAULT",
    "resnet50": "ResNet50_Weights.DEFAULT",
}


def save_transformed_samples(dataloader):
    sz = 200
    denormalize = T.Compose([
        T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]),
        T.Resize(sz, antialias=True)
    ])
    batch, label = next(iter(dataloader))
    batch = batch[0] if isinstance(batch,
                                   Sequence) else batch  # in case batch comes from TwoStreamBatchSampler

    R, C, ix = 5, 5, 0
    image_samples = np.zeros((batch.shape[1], sz * C, sz * R))
    for row in range(R):
        for col in range(C):
            img = denormalize(batch[ix]).cpu()
            image_samples[:, col * sz: (col + 1) * sz, row * sz: (row + 1) * sz] = img
            ix += 1
            if ix >= len(batch):
                ix = 0
                batch, label = next(iter(dataloader))
                batch = batch[0] if isinstance(batch, Sequence) else batch

    save_image(os.path.join(result_subdir, 'training_samples.png'), image_samples)


def get_run_desc():
    run_desc = args.arch

    if args.evaluate:
        run_desc += '_evaluate'
    else:
        run_desc += '_finetune'
        run_desc += "_{}_{}".format(args.ftune_head, args.ftune_strategy)

        if args.all_labels:
            num_labels_str = 'all'
        elif args.num_labels < 1:
            num_labels_str = "{}p".format(int(args.num_labels * 100))
        elif (args.num_labels % 1000) == 0:
            num_labels_str = '%dk' % (args.num_labels / 1000)
        else:
            num_labels_str = '%d' % args.num_labels

        if args.exclude_unlabeled:
            learning_stra = "sv"  # supervised
        elif args.ss_method == "mean-teacher":
            learning_stra = "mt"
        elif args.ss_method == "mixmatch":
            learning_stra = "mm"
        elif args.ss_method == "VAT":
            learning_stra = "vat"

        run_desc += ('_{}_{}_{}'.format(num_labels_str, learning_stra, args.random_seed))

    return run_desc


def main():
    # Fixed random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

    # Export run information.
    report.export_sources(os.path.join(result_subdir, 'src'))
    report.export_run_details(os.path.join(result_subdir, 'run.txt'), args)
    report.export_config(os.path.join(result_subdir, 'config.txt'), args)

    # Get number of classes from train directory
    traindir = os.path.join(args.data, 'train')
    num_classes = len([name for name in os.listdir(traindir)])
    # get inference transforms from weight descriptor (refs.py [2,3])
    inference_transforms = models.get_weight(weight_descriptor[args.arch]).transforms(antialias=True)
    crop_sz = inference_transforms.crop_size[0]
    train_transforms = T.Compose([
        T.Resize(inference_transforms.resize_size),
        T.RandomCrop(crop_sz),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=inference_transforms.mean, std=inference_transforms.std),
    ])
    if args.train_transforms == "strong":
        train_transforms = T.Compose([
            T.RandomRotation(10),
            T.RandomResizedCrop(crop_sz),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=inference_transforms.mean, std=inference_transforms.std),
        ])
    print("Using '{}' training augmentations.".format(args.train_transforms))

    val_loader = DataLoader(ImageFolder(f"{args.data}/val", inference_transforms),
                            batch_size=64, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
    train_ds = ImageFolder(f"{args.data}/train", train_transforms)
    train_loader = None

    if args.all_labels:  # use all labels
        """ Supervised training """
        print("Keeping all labels.")

        train_loader = DataLoader(train_ds, batch_size=args.labeled_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        # Create model
        print("\n*** Supervised Only Training ***")
        print("Using pre-trained model '{}' with {} classes.".format(args.arch, num_classes))
        original_model = models.__dict__[args.arch](weights="DEFAULT")
        model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                              name="Supervised").to(device)
        finetuner = FineTunerSupervised(model, (train_loader, val_loader), args,
                                        result_subdir, device)
    else:
        labeled_idxs, unlabeled_idxs = split_idxs(train_ds, args.num_labels)
        print("Keeping {} of {} labels.".format(len(labeled_idxs), len(train_ds)))
        if args.exclude_unlabeled:
            """ Supervised training """
            print("Excluding unlabeled samples: creating SubsetRandomSampler.")

            sampler = SubsetRandomSampler(labeled_idxs)
            train_loader = DataLoader(train_ds, sampler=sampler, batch_size=args.labeled_batch_size,
                                      num_workers=args.num_workers, pin_memory=True)
            # Create model
            print("\n*** Supervised Only Training ***")
            print("Using pre-trained model '{}' with {} classes.".format(args.arch, num_classes))
            original_model = models.__dict__[args.arch](weights="DEFAULT")
            model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                                  name="Supervised").to(device)
            finetuner = FineTunerSupervised(model, (train_loader, val_loader), args, result_subdir,
                                            device)
        elif args.ss_method == "mean-teacher":
            """ Semi-supervised training: Mean Teacher """
            print("Keeping {} of {} unlabeled samples: creating TwoStreamBatchSampler.".format(
                len(unlabeled_idxs), len(train_ds)))

            train_ds.transform = TransformTwice(train_transforms)
            batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs,
                                                  args.aux_batch_size, args.labeled_batch_size)
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                      num_workers=args.num_workers, pin_memory=True)

            print("\n*** Semi-supervised Training: Mean Teacher ***")
            # Create student model
            print("Creating student model '{}' with {} classes.".format(args.arch, num_classes))
            original_model = models.__dict__[args.arch](weights="DEFAULT")
            model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                                  name="Student").to(device)
            # Create EMA model
            print("Creating teacher model '{}' with {} classes.".format(args.arch, num_classes))
            original_model = models.__dict__[args.arch](
                weights=None)  # No weights - random initialization
            ema_model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                                      name="Mean Teacher").to(device)
            ema_model.freeze_all()  # EMA model is not trainable
            finetuner = FineTunerMeanTeacher((model, ema_model), (train_loader, val_loader), args,
                                             result_subdir, device)
        elif args.ss_method == "mixmatch":
            """ Semi-supervised training: MixMatch """
            train_labeled_set = train_ds
            train_unlabeled_set = \
                ImageFolder(f"{args.data}/train",
                            transform=mixmatch.Augmentation(args.K, train_transforms))

            train_labeled_set = Subset(train_labeled_set, labeled_idxs)
            train_unlabeled_set = Subset(train_unlabeled_set, unlabeled_idxs)
            train_loader = \
                DataLoader(train_labeled_set, batch_size=args.labeled_batch_size,
                           num_workers=args.num_workers,
                           sampler=InfiniteSampler(len(train_labeled_set)))
            ul_train_loader = \
                DataLoader(train_unlabeled_set,
                           batch_size=(args.aux_batch_size - args.labeled_batch_size),
                           num_workers=args.num_workers,
                           sampler=InfiniteSampler(len(train_unlabeled_set)))

            print("\n*** Semi-supervised Training: MixMatch ***")
            # Create model
            print("Creating model '{}' with {} classes.".format(args.arch, num_classes))
            original_model = models.__dict__[args.arch](weights="DEFAULT")
            model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                                  name="Stu Model").to(device)

            # Create EMA model
            print("Creating EMA model '{}' with {} classes.".format(args.arch, num_classes))
            original_model = models.__dict__[args.arch](
                weights=None)  # No weights - random initialization
            ema_model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                                      name="EMA Model").to(device)
            ema_model.freeze_all()  # EMA model is not trainable
            finetuner = FineTunerMixMatch((model, ema_model),
                                          ((train_loader, ul_train_loader), val_loader), args,
                                          result_subdir, device)
        elif args.ss_method == "VAT":
            """ Semi-supervised training: VAT """
            train_labeled_set = Subset(train_ds, labeled_idxs)
            train_unlabeled_set = Subset(train_ds, unlabeled_idxs)
            train_loader = \
                DataLoader(train_labeled_set, batch_size=args.labeled_batch_size,
                           num_workers=args.num_workers,
                           sampler=InfiniteSampler(len(train_labeled_set)))
            ul_train_loader = \
                DataLoader(train_unlabeled_set,
                           batch_size=(args.aux_batch_size - args.labeled_batch_size),
                           num_workers=args.num_workers,
                           sampler=InfiniteSampler(len(train_unlabeled_set)))

            print("\n*** Semi-supervised Training: VAT ***")
            # Create model
            print("Creating model '{}' with {} classes.".format(args.arch, num_classes))
            original_model = models.__dict__[args.arch](weights="DEFAULT")
            model = FineTuneModel(original_model, args.arch, num_classes, args.ftune_head,
                                  name="VAT Model").to(device)

            finetuner = FineTunerVAT(model, ((train_loader, ul_train_loader), val_loader),
                                     args, result_subdir, device)
        else:
            raise "Unsupported semi-supervised learning method."

    summary(model, torch.zeros(1, 3, crop_sz, crop_sz).to(device), depth=5)

    cudnn.benchmark = True
    save_transformed_samples(train_loader)

    finetuner.finetune()
    pass


if __name__ == '__main__':
    args = parse_commandline_args()

    # Argument validation
    if args.evaluate:
        args.ftune_strategy = "evaluate"
    if args.all_labels:
        args.exclude_unlabeled = True

    run_desc = get_run_desc()

    # Create the result directory and basic run data.
    result_subdir = report.create_result_subdir(args.result_dir, run_desc)

    # Start dumping stdout and stderr into result directory.
    stdout_tap = Tap(sys.stdout)
    stderr_tap = Tap(sys.stderr)
    sys.stdout = stdout_tap
    sys.stderr = stderr_tap
    stdout_tap.set_file(open(os.path.join(result_subdir, 'stdout.txt'), 'wt', encoding='utf-8'))
    stderr_tap.set_file(open(os.path.join(result_subdir, 'stderr.txt'), 'wt', encoding='utf-8'))

    print("Starting up...")
    print("Saving results to", result_subdir)
    main()
    print("Exiting...")
