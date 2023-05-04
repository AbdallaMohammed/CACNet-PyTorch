import os
import config
import torch
import numpy as np

from PIL import Image


def save_checkpoint(checkpoint):
    print('=> Saving checkpoint')

    torch.save(checkpoint, config.CHECKPOINT)


def load_checkpoint(model, optimizer=None):
    print('=> Loading checkpoint')

    checkpoint = torch.load(config.CHECKPOINT)

    if 'state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)

        return 1

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def can_load_checkpoint():
    return os.path.exists(config.CHECKPOINT) and config.LOAD_MODEL


def generate_anchors(anchor_stride):
    P_h = np.array([2 + i * 4 for i in range(16 // anchor_stride)])
    P_w = np.array([2 + i * 4 for i in range(16 // anchor_stride)])

    num_anchors = len(P_h) * len(P_h)

    anchors = torch.zeros((num_anchors, 2))
    k = 0

    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k, 1] = P_w[j]
            anchors[k, 0] = P_h[i]

            k += 1

    return anchors


def shift(shape, stride, anchors):
    shift_w = torch.arange(0, shape[0]) * stride
    shift_h = torch.arange(0, shape[1]) * stride

    shift_w, shift_h = torch.meshgrid([shift_w, shift_h])

    shifts = torch.stack([shift_w, shift_h], dim=-1)

    trans_anchors = anchors.unsqueeze(1).unsqueeze(2)
    trans_shifts = shifts.unsqueeze(0)

    anchors = trans_anchors + trans_shifts

    return anchors


def compute_iou(y_predict, y_true):
    zeros  = torch.zeros(y_true.shape[0])

    x1 = torch.maximum(y_true[:, 0], y_predict[:, 0])
    y1 = torch.maximum(y_true[:, 1], y_predict[:, 1])
    x2 = torch.minimum(y_true[:, 2], y_predict[:, 2])
    y2 = torch.minimum(y_true[:, 3], y_predict[:, 3])

    w  = torch.maximum(zeros, x2 - x1)
    h  = torch.maximum(zeros, y2 - y1)

    intercetion = w * h

    area1 = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area2 = (y_predict[:, 2] - y_predict[:, 0]) * (y_predict[:, 3] - y_predict[:, 1])

    union = area1 + area2 - intercetion
    iou = intercetion / union

    index = iou.argmax(-1)

    return iou[index].item()


def apply_cacnet(image, model):
    image = Image.open(image).convert('RGB')
        
    # Original Size
    width, height = image.size

    batch = config.DATASETS['FCDB']['TRANSFORMS'](image=np.array(image))['image']
    batch = batch.to(config.DEVICE)

    _, _, boxes = model(batch.unsqueeze(0))

    boxes[:,0::2] = boxes[:,0::2] / 224 * width
    boxes[:,1::2] = boxes[:,1::2] / 224 * height

    boxes = boxes.cpu().detach().numpy()
    x1, y1, x2, y2 = boxes[0].astype(np.int32)

    best_crop = Image.fromarray(np.asarray(image)[y1:y2, x1:x2]).convert('RGB')

    return best_crop
