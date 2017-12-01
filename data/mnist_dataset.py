import torch
import torchvision
import numpy as np
from collections import deque
from .data_affine import RandomAffineTransform

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, D=5, train=True, transform=None, target_transform=None, download=False, mnist_directory="/data/sls/scratch/kkleidal/mnist"):
        self.D = D
        self.transform = transform
        self.mnist = torchvision.datasets.MNIST(mnist_directory, train=train, target_transform=target_transform,
                download=download)
        self.labels = []
        self.label_map = {}
        for i, (_, label) in enumerate(self.mnist):
            self.labels.append(label)
            if label not in self.label_map:
                self.label_map[label] = []
            self.label_map[label].append(i)
        self.labels = np.array(self.labels)
        self.label_map = {label: np.array(lst) for label, lst in self.label_map.items()}
        self.conditional_buffer = {label: deque() for label in self.label_map}

    def __len__(self):
        return len(self.mnist)

    def _apply_transform(self, img):
        img = np.array(img)
        img = np.expand_dims(img, -1)
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            for transform in self.transform:
                img = transform(img)
        img = np.transpose(img, [2, 0, 1])
        return img

    def _get_conditional(self, label, count):
        while len(self.conditional_buffer[label]) < count:
            self.conditional_buffer[label].extend(np.random.permutation(self.label_map[label]))
        Didx = [self.conditional_buffer[label].popleft() for _ in range(count)]
        D = [self._apply_transform(self.mnist[idx][0]) for idx in Didx]
        D = np.stack(D, axis=0)
        return D

    def __getitem__(self, idx):
        x_image, x_label = self.mnist[idx]
        x_image = self._apply_transform(x_image)
        D = self._get_conditional(x_label, self.D)
        return x_label, x_image, D

if __name__ == "__main__":
    ds = MnistDataset(download=True, transform=[RandomAffineTransform()])
    label, x, D = ds[0]
    
    print(label)
    print(x.shape)
    print(D.shape)
