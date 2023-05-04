import torch
import config
import utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from models import CACNet
from dataset import KUPCPDataset, FCDBDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


def train_epoch(fcdb_loader, kupcp_loader, model, cropper_loss_fn, classifier_loss_fn, optimizer, epoch, lr_scheduler=None):
    model.train()

    loader = tqdm(fcdb_loader)

    cropper_losses = []
    classifier_losses = []
    classifier_accuracies = []

    for boxes, images, width, height in loader:
        # Train Cropper Branch
        boxes = boxes.to(config.DEVICE).squeeze(1)
        images = images.to(config.DEVICE)
        height = height.to(config.DEVICE)
        width = width.to(config.DEVICE)

        boxes[:, 0::2] = boxes[:, 0::2] / width[:, None] * images.shape[-1]
        boxes[:, 1::2] = boxes[:, 1::2] / height[:, None] * images.shape[-2]

        model.classifier.eval()

        _, _, target_boxes = model(images)

        cropper_loss = cropper_loss_fn(target_boxes, boxes)

        optimizer.zero_grad()
        cropper_loss.backward()

        cropper_losses.append(cropper_loss.item())

        # Train Classifier Branch
        labels, images = next(iter(kupcp_loader))

        labels = labels.to(config.DEVICE)
        images = images.to(config.DEVICE)

        model.classifier.train()

        scores, _, _ = model(images)
        _, predictions = torch.max(scores, dim=1)

        classifier_accuracy = torch.mean((predictions == labels).float()) * 100

        classifier_loss = classifier_loss_fn(scores, labels)

        classifier_loss.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        loader.set_postfix(cropper_loss=cropper_loss.item(), classifier_loss=classifier_loss.item(), classifier_accuracy=classifier_accuracy.item())

        classifier_losses.append(classifier_loss.item())
        classifier_accuracies.append(classifier_accuracy.item())

    if config.SAVE_MODEL:
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'cropper_losses': np.mean(cropper_losses),
            'classifier_losses': np.mean(classifier_losses)
        })

    print(f'Epoch[{epoch}] => Classifier Loss [{np.mean(classifier_losses):.2f}], Cropper Loss [{np.mean(cropper_losses):.2f}], Classifier Accuracy [{np.mean(classifier_accuracies):.2f}]')


if __name__ == '__main__':
    fcdb_dataset = FCDBDataset(
        root=config.DATASETS['FCDB']['DATASET'],
        annotations=config.DATASETS['FCDB']['ANNOTATIONS']['TRAIN'],
        transform=config.DATASETS['FCDB']['TRANSFORMS'],
        augmentation=True,
    )

    fcdb_loader = DataLoader(
        dataset=fcdb_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        shuffle=True,
    )

    kupcp_dataset = KUPCPDataset(
        root=config.DATASETS['KUPCP']['TRAIN'],
        root_labels=config.DATASETS['KUPCP']['LABELS']['TRAIN'],
        transform=config.DATASETS['KUPCP']['TRANSFORMS']['TRAIN'],
    )

    kupcp_loader = DataLoader(
        dataset=kupcp_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        shuffle=True,
    )

    model = CACNet()
    model = model.to(config.DEVICE)

    cropper_loss_fn = nn.SmoothL1Loss()
    classifier_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.LR_DECAY_EPOCH, gamma=config.LR_DECAY)

    starting_epoch = 1

    if utils.can_load_checkpoint():
        utils.load_checkpoint(model)

    for epoch in range(starting_epoch, config.EPOCHS):
        train_epoch(
            fcdb_loader,
            kupcp_loader,
            model,
            cropper_loss_fn,
            classifier_loss_fn,
            optimizer,
            epoch,
            lr_scheduler,
        )