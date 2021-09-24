import numpy as np
import torch
from torchvision import transforms

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CaptchaDataset:
    def __init__(self, image_paths, targets, resize=None):
        # resize = (height, width)
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.targets[idx]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)
        image = self.transform(image)

        return image, torch.Tensor(target)
