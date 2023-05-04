import config
import utils
import numpy as np

from tqdm import tqdm
from models import CACNet
from dataset import FCDBDataset, KUPCPDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

 
def eval_model(loader, model):
    model.eval()

    scores = []

    for boxes, images, width, height in tqdm(loader):
        boxes = boxes.to(config.DEVICE).squeeze(1)
        images = images.to(config.DEVICE)
        height = height.to(config.DEVICE)
        width = width.to(config.DEVICE)

        boxes[:, 0::2] = boxes[:, 0::2] / width[:, None] * images.shape[-1]
        boxes[:, 1::2] = boxes[:, 1::2] / height[:, None] * images.shape[-2]

        _, _, target_boxes = model(images)

        scores.append(
            utils.compute_iou(
                boxes.cpu(),
                target_boxes.cpu()
            )
        )
    
    print(f'Eval IoU [{(np.mean(scores) * 100):.2f}]')


if __name__ == '__main__':
    fcdb_dataset = FCDBDataset(
        root=config.DATASETS['FCDB']['DATASET'],
        annotations=config.DATASETS['FCDB']['ANNOTATIONS']['TEST'],
        transform=config.DATASETS['FCDB']['TRANSFORMS'],
        augmentation=False,
    )
    
    fcdb_loader = DataLoader(
        dataset=fcdb_dataset,
        pin_memory=config.PIN_MEMORY,
        batch_size=config.BATCH_SIZE,
        drop_last=False,
        shuffle=False,
    )

    kupcp_dataset = KUPCPDataset(
        root=config.DATASETS['KUPCP']['TEST'],
        root_labels=config.DATASETS['KUPCP']['LABELS']['TEST'],
        transform=config.DATASETS['KUPCP']['TRANSFORMS']['TEST'],
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

    if utils.can_load_checkpoint():
        utils.load_checkpoint(model)

    eval_model(
        fcdb_loader,
        kupcp_loader,
        model
    )
