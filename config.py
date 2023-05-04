import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2


DATASETS = {
    'KUPCP': {
        'TRAIN': './dataset/KUPCP/train',
        'TEST': './dataset/KUPCP/test',
        'LABELS': {
            'TRAIN': './dataset/KUPCP/labels/train.txt',
            'TEST': './dataset/KUPCP/labels/test.txt',
        },
        'TRANSFORMS': {
            'TRAIN': A.Compose([
                A.Resize(height=224, width=224),
                A.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]),
            'TEST': A.Compose([
                A.Resize(height=224, width=224),
                A.Normalize(),
                ToTensorV2(),
            ])
        },
    },
    'FCDB': {
        'DATASET': './dataset/FCDB/data',
        'ANNOTATIONS': {
            'TRAIN': './dataset/FCDB/annotations/cropping_training_set.json',
            'TEST': './dataset/FCDB/annotations/cropping_testing_set.json'
        },
        'TRANSFORMS': A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2(),
        ])
    },
}

LABELS_TO_INT_MAP = {
    'rot': 0,
    'center': 1,
    'horizontal': 2,
    'symmetric': 3,
    'diagonal': 4,
    'curved': 5,
    'vertical': 6,
    'triangle': 7,
    'repeated_pattern': 8,
}

CHECKPOINT = './checkpoints/ccnet.pth.tar'

IMAGE_SIZE = 224

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
PIN_MEMORY = False

ANCHOR_STRIDE = 8
LEARNING_RATE = 3.5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 80

LR_DECAY_EPOCH = [30, 60]
LR_DECAY = 0.1

LOAD_MODEL = True
SAVE_MODEL = True
