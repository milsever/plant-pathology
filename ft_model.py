"""
Author: Murat Ilsever
Date: 2024-03-16
Description: Creates a classification model to be fine-tuned.
"""

import copy
import torch.nn as nn
import torch.nn.functional as F


class FineTuneModel(nn.Module):
    """Creates a fine-tune model compatible with number of classes.
       All parameters are frozen except head.
       Head model can be a simple softmax or a FCN.
    """

    # head_model: ['softmax', 'fcn']
    def __init__(self, original_model, arch, num_classes, head_model='softmax', name=""):
        super(FineTuneModel, self).__init__()
        self.trainable = 'head'
        self.name = name
        self.num_classes = num_classes
        self.arch = arch

        if arch in ['resnet50', 'inception_v3']:
            self.features = copy.deepcopy(original_model)
            self.features.fc = nn.Identity()
            num_ftrs = original_model.fc.in_features
            if head_model == 'softmax':
                self.classifier = self._get_softmax_head(num_ftrs, num_classes)
            else:
                self.classifier = self._get_fcn_head(num_ftrs, num_classes)

        elif arch in ['densenet121', 'densenet201']:
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            num_ftrs = original_model.classifier.in_features
            if head_model == 'softmax':
                self.classifier = self._get_softmax_head(num_ftrs, num_classes)
            else:
                self.classifier = self._get_fcn_head(num_ftrs, num_classes)

        elif arch in ['efficientnet_b3', 'efficientnet_v2_m', 'efficientnet_v2_s']:
            self.features = copy.deepcopy(original_model.features)
            num_ftrs = original_model.classifier[1].in_features
            if head_model == 'softmax':
                self.classifier = self._get_softmax_head(num_ftrs, num_classes)
            else:
                self.classifier = self._get_fcn_head(num_ftrs, num_classes)

        elif arch in ['convnext_tiny']:
            self.features = copy.deepcopy(original_model.features)
            num_ftrs = original_model.classifier[2].in_features
            if head_model == 'softmax':
                self.classifier = self._get_softmax_head(num_ftrs, num_classes)
            else:
                self.classifier = self._get_fcn_head(num_ftrs, num_classes)

        elif arch in ['mobilenet_v3_large']:
            self.features = copy.deepcopy(original_model.features)
            num_ftrs = original_model.classifier[0].in_features
            if head_model == 'softmax':
                self.classifier = self._get_softmax_head(num_ftrs, num_classes)
            else:
                self.classifier = self._get_fcn_head(num_ftrs, num_classes)

        else:
            raise 'Fine-tuning not supported on this architecture yet.'

        self.features.leaf_modules = []
        for module in self.features.modules():
            if len(list(module.children())) == 0:
                self.features.leaf_modules.append(module)

        # start body frozen
        self.freeze_body()

    def forward(self, x):
        feat = self.features(x)
        if self.arch in ['densenet121', 'densenet201', 'efficientnet_b3', 'efficientnet_v2_m', 'efficientnet_v2_s',
                         'convnext_tiny', 'mobilenet_v3_large']:
            feat = F.adaptive_avg_pool2d(feat, output_size=1)
            feat = feat.view(feat.size(0), -1)
        elif self.arch in ['resnet50', 'inception_v3']:
            if type(feat).__name__ == "InceptionOutputs":
                feat = feat.logits

        y = self.classifier(feat)
        return y

    @staticmethod
    def _get_fcn_head(num_ftrs, num_classes):
        return nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    @staticmethod
    def _get_softmax_head(num_ftrs, num_classes):
        return nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )

    # freeze network body parameters
    def freeze_body(self):
        self.trainable = 'head'
        print('Freezing all body parameters. (Model: {}, Arch: {})'.format(
            self.name, self.arch.upper()))
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_body(self):
        self.trainable = 'all'
        print('Unfreezing body parameters except BN modules. (Model: {}, Arch: {})'.format(
            self.name, self.arch.upper()))
        for module in self.features.leaf_modules:
            if not isinstance(module, nn.BatchNorm2d):
                for name, param in module.named_parameters():
                    param.requires_grad = True

    # freeze all network parameters
    def freeze_all(self):
        print('Freezing all parameters. (Model: {}, Arch: {})'.format(self.name, self.arch.upper()))
        for p in self.parameters():
            p.requires_grad = False
