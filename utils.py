import os
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")  # Use CPU

print("device: ", device)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(image):
    image = image.convert("RGB")
    # Preprocessing transforms
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return preprocess(image)


def preprocess_val_image(image):
    image = image.convert("RGB")
    # Preprocessing transforms
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return preprocess(image)


num_images_in_phase = [
    294,
    299,
    293,
    292,
    297,
    299,
    297,
    298,
    293,
    298,
    887,
]

NUM_PHASES = len(num_images_in_phase)
MEMORY_SIZE = 5
NUM_CLASSES_IN_PHASE = 10


# Create Custom PyTorch Dataset
class CustomImageDataset(Dataset):
    def __init__(self, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.img_labels = []
        self.image_paths = []

        if phase <= 0:
            data_dir = f"data/Val"
            # print("data_dir: ", data_dir)
            for k in range(num_images_in_phase[-1]):
                image_filename = f"{k:03d}.jpg"
                image_path = os.path.join(data_dir, image_filename)

                # print("image_path: ", image_path)

                # Check if the image file exists
                if os.path.exists(image_path):
                    self.img_labels.append(image_filename)
                    self.image_paths.append(image_path)
                else:
                    print("not found: ", image_path)
        else:
            for i in range(phase):
                for j in range(NUM_CLASSES_IN_PHASE):
                    label = i * NUM_CLASSES_IN_PHASE + j
                    if i < phase - 1:
                        data_dir = f"data/Train/phase_{i + 1}/{label:03d}"
                        # print("data_dir: ", data_dir)
                        for k in range(MEMORY_SIZE):
                            self.img_labels.append(label)

                            image_filename = f"{k:03d}.jpg"
                            image_path = os.path.join(data_dir, image_filename)
                            self.image_paths.append(image_path)
                    else:
                        data_dir = f"data/Train/phase_{i + 1}/{label:03d}"
                        # print("data_dir: ", data_dir)
                        for k in range(num_images_in_phase[i - 1]):
                            image_filename = f"{k:03d}.jpg"
                            image_path = os.path.join(data_dir, image_filename)

                            # print("image_path: ", image_path)

                            # Check if the image file exists
                            if os.path.exists(image_path):
                                self.img_labels.append(label)
                                self.image_paths.append(image_path)
                            else:
                                break

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        image_path = self.image_paths[idx]
        # print("image_path: ", image_path)

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class NeuralNetwork(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext101_32x8d(weights=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes),
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        output = self.base_model(x)
        return self.sigm(output)


# save checkpoint function
def checkpoint_save(model, save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    torch.save(model.state_dict(), f)
    print("saved checkpoint:", filename)


# save checkpoint function
def checkpoint_load(model, save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    model.load_state_dict(torch.load(f))
    print("loaded checkpoint:", filename)


# calculate metrics function
def calculate_metrics(pred, target):
    pred = [np.argmax(i) for i in pred]
    return {
        "weighted/f1": f1_score(y_true=target, y_pred=pred, average="weighted"),
    }