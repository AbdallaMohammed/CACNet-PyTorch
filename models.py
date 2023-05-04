import torch
import config
import utils
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.model = models.vgg16(weights='DEFAULT')
        self.features = self.model.features
        
        self.features1 = self.features[:6]
        self.features2 = self.features[6:10]
        self.features3 = self.features[10:17]
        self.features4 = self.features[17:30]


    def forward(self, images):
        with torch.no_grad():
            f1 = self.features1(images)
            f2 = self.features2(f1)
            f3 = self.features3(f2)
            f4 = self.features4(f3)

        return f2, f3, f4


class Classifier(nn.Module):
    def __init__(self, num_classes=9):
        super(Classifier, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
        )

        self.fc = nn.Linear(128, self.num_classes)

    def forward(self, f2, f3, f4):
        features = self.conv1(f4)
        features = self.conv2(features)

        features = F.interpolate(features, scale_factor=2, mode='bilinear') + f3

        features = self.conv3(features)

        features = F.interpolate(features, scale_factor=2, mode='bilinear') + f2

        # GAP section
        fc = torch.flatten(F.adaptive_avg_pool2d(features, 1), start_dim=1)
        scores = F.softmax(self.fc(fc), dim=1)

        # Compute CAMs and KCMs
        with torch.no_grad():
            # Compute CAMs
            batch, composition, height, width = features.shape
            features = features.permute(0, 1, 2, 3)
            features = features.view(batch, composition, height * width)

            w = self.fc.weight.data.unsqueeze(0).repeat(batch, 1, 1)

            cams = torch.matmul(w, features) # (batch, compisitions, height * width)

            # Compute KCMs
            cams = self._normalize_cams(cams)
            cams = cams.view(batch, self.num_classes, height, width) # (batch, 9, height, width)

            kcms = torch.sum(scores[:, :, None, None] * cams, dim=1, keepdim=True)
            kcms = F.interpolate(kcms, scale_factor=4, mode='bilinear', align_corners=True)

        return fc, cams, kcms
    
    def _normalize_cams(self, cam):
        cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
        cam = cam / cam.max(dim=-1)[0].unsqueeze(-1)

        return cam


class Cropper(nn.Module):
    def __init__(self, anchor_stride):
        super(Cropper, self).__init__()

        self.out_features = int((16 / anchor_stride)**2 * 4)
        self.num_anchors = (16 // anchor_stride)**2
        self.upscale_factor = self.num_anchors // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(
            256,
            self.out_features,
            kernel_size=(3, 3),
            padding=1
        )

        anchors = utils.generate_anchors(anchor_stride)
        feat_shape = (config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)
        
        all_anchors = utils.shift(feat_shape, 16, anchors)
        all_anchors = all_anchors.float().unsqueeze(0)

        anchors_x = F.pixel_shuffle(all_anchors[..., 0], upscale_factor=self.upscale_factor)
        anchors_y = F.pixel_shuffle(all_anchors[..., 1], upscale_factor=self.upscale_factor)

        all_anchors = torch.stack([anchors_x, anchors_y], dim=-1).squeeze(1)

        grid_x = (all_anchors[..., 0] - config.IMAGE_SIZE / 2) / (config.IMAGE_SIZE / 2)
        grid_y = (all_anchors[..., 1] - config.IMAGE_SIZE / 2) / (config.IMAGE_SIZE / 2)

        grid = torch.stack([grid_x, grid_y], dim=-1)

        self.register_buffer('all_anchors', all_anchors)
        self.register_buffer('grid', grid)

    def forward(self, features, kcms):
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)

        anchors = self.output(features)

        batch, _, height, width = anchors.shape
        anchors = anchors.view(batch, self.num_anchors, height, width, 4)

        coords = [F.pixel_shuffle(anchors[..., i], upscale_factor=self.upscale_factor) for i in range(4)]

        offsets = torch.stack(coords, dim=-1).squeeze(1)

        regression = torch.zeros_like(offsets)
        regression[..., 0::2] = offsets[..., 0::2] + self.all_anchors[..., 0:1]
        regression[..., 1::2] = offsets[..., 1::2] + self.all_anchors[..., 1:2]

        trans_grid = self.grid.repeat(offsets.shape[0], 1, 1, 1)

        sample_kcm = F.grid_sample(kcms, trans_grid, mode='bilinear', align_corners=True)
        reg_weight = F.softmax(sample_kcm.flatten(1), dim=1).unsqueeze(-1)

        batch, height, width, compesition = regression.shape
        regression = regression.view(batch, height * width, compesition)

        return torch.sum(reg_weight * regression, dim=1)


class CACNet(nn.Module):
    def __init__(self):
        super(CACNet, self).__init__()

        self.vgg16 = VGG16()
        self.classifier = Classifier()
        self.cropper = Cropper(anchor_stride=config.ANCHOR_STRIDE)

    def forward(self, images):
        f2, f3, f4 = self.vgg16(images)

        scores, _, kcms = self.classifier(f2, f3, f4)

        boxes = self.cropper(f4, kcms)

        return scores, kcms, boxes
