import os
import json
import config
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class KUPCPDataset(Dataset):
    def __init__(self, root, root_labels, transform=None):
        self.root = root
        self.transform = transform
        self.annotations = []

        image_list = os.listdir(self.root)
        image_list = sorted(image_list, key=lambda k: float(os.path.splitext(k)[0]))

        image_idx  = 0
        txt_idx    = 0

        with open(root_labels, 'r') as f:
            for line in f.readlines():
                txt_idx += 1
                image_name = image_list[image_idx]

                if txt_idx < int(os.path.splitext(image_name)[0]):
                    continue

                image_idx += 1

                labels = line.strip().split(' ')
                labels = [int(l) for l in labels]

                categories = [i for i in range(len(labels)) if labels[i] == 1]
                
                if len(categories) > 0:
                    self.annotations.append((image_name, categories))

    def __getitem__(self, item):
        image, label = self.annotations[item]

        image = np.array(Image.open(os.path.join(self.root, image)).convert('RGB'))

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return torch.tensor(label[-1], dtype=torch.long), image

    def __len__(self):
        return len(self.annotations)
    

class FCDBDataset(Dataset):
    def __init__(self, root, annotations, augmentation=False, transform=None):
        self.root = root
        self.transform = transform
        self.augmentation = augmentation

        self.images = []
        self.annotation = {}

        annotations = json.loads(open(annotations, 'r').read())

        for item in annotations:
            url = item['url']
            image_name = os.path.split(url)[-1]
            
            if os.path.exists(os.path.join(self.root, image_name)):
                x, y, w, h = item['crop']
                crop = [x, y, x + w, y + h]

                self.annotation[image_name] = crop
                self.images.append(image_name)

    def __getitem__(self, item):
        image_name = self.images[item]

        image = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        box = np.array(self.annotation[image_name]).reshape(-1, 4).astype(np.float32)

        width, height = image.size
        
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return box, image, width, height

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    kupcp_dataset = KUPCPDataset(
        root=config.DATASETS['KUPCP']['TRAIN'],
        root_labels=config.DATASETS['KUPCP']['LABELS']['TRAIN'],
        transform=config.DATASETS['KUPCP']['TRANSFORMS']['TRAIN'],
    )

    fcdb_dataset = FCDBDataset(
        root=config.DATASETS['FCDB']['DATASET'],
        annotations=config.DATASETS['FCDB']['ANNOTATIONS']['TRAIN'],
        transform=config.DATASETS['FCDB']['TRANSFORMS'],
        augmentation=True,
    )

    print(len(kupcp_dataset), len(fcdb_dataset))
