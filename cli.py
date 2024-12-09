"""
Author: Murat Ilsever
Date: 2022-12-15
Description: Command line arguments for train.py.
"""

import argparse
import textwrap
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


def create_parser():
    parser = argparse.ArgumentParser(description='Plant Pathology Training',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--result-dir', default='result', type=str, metavar='DIR',
                        help='directory for storing the results of individual training runs')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    # parser.add_argument('--finetune', action='store_true',
    #                     help='fine tune pre-trained model')
    parser.add_argument('--ftune-head', metavar='HEAD', default='softmax', choices=['softmax', 'fcn'],
                        help=textwrap.dedent('''\
                        softmax: single linear layer connecting last linear output to num_classes units
                        fcn: head model with series of FC => RELU => DROPOUT layers 
                        (default: simple)
                        '''))
    parser.add_argument('--ftune-strategy', metavar='ST', default='headonly',
                        choices=['headonly', 'headthenbody', 'headandbody'],
                        help=textwrap.dedent('''\
                        headonly: freeze all layers in the body of the network and train the layer head
                        headthenbody: first train the layer head, then unfreeze the body and train that, too
                        headandbody: simply leave all layers unfrozen and train them all together
                        (default: headonly)
                        '''))
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help='evaluate model on validation set')
    # parser.add_argument('--weight-descriptor', metavar='DESC', default='ResNet18_Weights.IMAGENET1K_V1',
    #                     help='enum value defining model weights (default: ResNet18_Weights.IMAGENET1K_V1)')
    parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--all-labels', action='store_true',
                        help='use all labels')
    parser.add_argument('--num-labels', default=400, type=float, metavar='N',
                        help=textwrap.dedent('''\
                        amount of labeled samples. if this number is greater than 1, it is the number of labeled 
                        samples. if it is in the range [0 1], it is the percentage of data that will be used as 
                        labeled. (default= 400)
                        '''))
    parser.add_argument('--exclude-unlabeled', action='store_true',
                        help='exclude unlabeled samples from the training set')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run in both head training and fine-tuning.')
    parser.add_argument('--early-stopping', action='store_true',
                        help='use early stopping')
    parser.add_argument('--patience', default=5, type=int, metavar='N',
                        help='when using early-stopping determines how long to wait after last time validation loss improved.')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='epoch to start at (useful on restarts)')
    parser.add_argument('--labeled-batch-size', default=32, type=int, metavar='N',
                        help="labeled samples per minibatch (default: 32)")
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                        help='learning rate used during head training (default: 0.01)')
    parser.add_argument('--fine-tuning-lr', default=1e-4, type=float, metavar='LR',
                        help='learning rate used during fine-tuning (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--train-transforms', metavar='TRANS', default='weak', choices=['weak', 'strong'],
                        help=textwrap.dedent('''\
                        weak: Resize, RandomCrop, RandomHorizontalFlip
                        strong: RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ColorJitter
                        (default: weak)
                        '''))
    # Semi-supervised Methods
    parser.add_argument('--ss-method', default='mean-teacher', choices=['mean-teacher', 'mixmatch', 'VAT'],
                        help=textwrap.dedent('''\
                        '''))
    # Mean teacher args
    parser.add_argument('--aux-batch-size', default=128, type=int, metavar='N',
                        help=textwrap.dedent('''\
                        size of a batch composed of labeled and unlabeled samples in mean teacher training.
                        labeled-batch-size defines number of labeled samples in a batch, rest is unlabeled.
                        (semi-supervised only) (default: 128)
                        '''))
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='teacher model decay rate (mean-teacher only) (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='final consistency loss weight after ramp-up (mean-teacher only) (default: None)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up in epochs (default: 30)')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')

    # MixMatch args
    parser.add_argument('--num-iters', default=400, type=int, metavar='N',
                        help="number of iterations per epoch (default: 400)")
    parser.add_argument('-K', default=2, type=int, metavar='K',
                        help="number of unlabeled augmentations (default: 2)")
    parser.add_argument('-T', default=0.5, type=float, metavar='T',
                        help='sharpening temperature (default: 0.5)')
    parser.add_argument('--lambda_u', default=75, type=float, metavar='LAM_U',
                        help='unsupervised loss weight for MixMatch (default: 75)')
    parser.add_argument('--alpha', default=0.75, type=float, metavar='ALPHA',
                        help='beta distribution parameter for MixMatch (default: 0.75)')
    # --ema-alpha <== --ema-decay

    # VAT args
    parser.add_argument('--epsilon', type=float, default=2.5)
    # num-iters

    # Other args
    parser.add_argument('--cuda-device-num', default=0, type=int, metavar='N',
                        help='which GPU to use (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--head-checkpoint', default='', type=str, metavar='PATH',
                        help='path to best head checkpoint (default: none)')
    parser.add_argument('-sf', '--snapshot-freq', default=0, type=int, metavar='N',
                        help='snapshot frequency in epochs (default: 5)')
    parser.add_argument('-pf', '--print-freq', default=10, type=int, metavar='N',
                        help='print frequency in iterations (default: 10)')
    parser.add_argument('--random-seed', default=1000, type=int, metavar='SEED',
                        help='randomization seed (default: 1000)')
    return parser


def parse_commandline_args():
    return create_parser().parse_args()
