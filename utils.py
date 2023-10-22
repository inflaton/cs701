import os
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")  # Use CPU

print("device: ", device)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# def preprocess_image(image):
#     image = image.convert("RGB")
#     # Preprocessing transforms
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize((256, 256)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#         ]
#     )
#     return preprocess(image)


# def preprocess_val_image(image):
#     image = image.convert("RGB")
#     # Preprocessing transforms
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#         ]
#     )
#     return preprocess(image)

# Initialize transformations for data augmentation
# https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
preprocess_image = transforms.Compose(
    [
        transforms.RandomResizedCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

preprocess_val_image = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


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

NUM_PHASES = len(num_images_in_phase) - 1
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


FC_LINEAR_ONE_OUT_FEATURES = 500


class NeuralNetwork(nn.Module):
    def __init__(self, n_classes, model_ver=1):
        super().__init__()
        self.model_ver = model_ver
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

        # for name, param in resnet.named_parameters():
        #     if "bn" not in name:
        #         param.requires_grad = False

        self.resnet_fc_in_features = resnet.fc.in_features
        resnet.fc = self.new_fc_layer(n_classes)
        self.n_classes = n_classes
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        output = self.base_model(x)
        return self.sigm(output)

    def new_fc_layer(self, n_classes):
        if self.model_ver == 1:
            return nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(
                    in_features=self.resnet_fc_in_features, out_features=n_classes
                ),
            )
        return nn.Sequential(
            nn.Linear(
                in_features=self.resnet_fc_in_features,
                out_features=FC_LINEAR_ONE_OUT_FEATURES,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=FC_LINEAR_ONE_OUT_FEATURES, out_features=n_classes),
        )

    def update_num_classes(self, n_classes):
        self.n_classes = n_classes
        self.base_model.fc[-1] = nn.Linear(
            in_features=FC_LINEAR_ONE_OUT_FEATURES, out_features=n_classes
        )


# save checkpoint function
def checkpoint_save(model, save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    torch.save(model.state_dict(), f)
    print("saved checkpoint:", f)


def checkpoint_delete(save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    os.remove(f)


# save checkpoint function
def checkpoint_load(model, save_path, epoch, n_classes=0):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)

    backup = None
    if n_classes > 0 and n_classes != model.n_classes:
        backup = model.n_classes
        model.update_num_classes(n_classes)

    model.load_state_dict(torch.load(f))

    if backup is not None:
        model.update_num_classes(backup)

    print("loaded checkpoint:", f)


# calculate metrics function
def calculate_metrics(pred, target):
    pred = [np.argmax(i) for i in pred]
    return {
        "weighted/f1": f1_score(y_true=target, y_pred=pred, average="weighted"),
    }


class TrainingImageDataset(Dataset):
    def __init__(self, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.img_labels = []
        self.image_paths = []

        for i in range(NUM_CLASSES_IN_PHASE):
            label = (phase - 1) * NUM_CLASSES_IN_PHASE + i
            data_dir = f"data/Train/phase_{phase}/{label:03d}"

            # print("data_dir: ", data_dir)
            for k in range(num_images_in_phase[phase - 1]):
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


trainingImageDatasets = [
    TrainingImageDataset(phase + 1, transform=preprocess_image)
    for phase in range(NUM_PHASES)
]


def get_training_datasets(phase, prev_train_sets=[], prev_val_sets=[]):
    dataset = trainingImageDatasets[phase - 1]

    train_len = int(len(dataset) * 7 / 10)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])

    counts = dict()
    for prev_train_set in prev_train_sets:
        indices = []
        for i in range(len(prev_train_set)):
            _, label = prev_train_set[i]
            if label not in counts:
                counts[label] = 1
                indices.append(i)
            elif counts[label] < MEMORY_SIZE:
                indices.append(i)
                counts[label] += 1

        temp_subset = Subset(prev_train_set, indices)
        train_set = ConcatDataset([train_set, temp_subset])

    if len(prev_val_sets) > 0:
        val_set = ConcatDataset([val_set, prev_val_sets[-1]])

    return train_set, val_set


def get_memory_subset(dataset, ids, max_count):
    counts = dict()
    indices = []
    for id in ids:
        _, label = dataset[id]

        if label not in counts:
            counts[label] = 1
            indices.append(id)
        elif counts[label] < max_count:
            indices.append(id)
            counts[label] += 1

    return Subset(dataset, indices)


# K-Fold Cross-Validation
kfoldSplits = []


def get_k_fold_training_datasets(phase, fold):
    if len(kfoldSplits) == 0:
        kfold = KFold(n_splits=MEMORY_SIZE, shuffle=True)

        for dataset in trainingImageDatasets:
            kfold_split_ids = []
            for train_ids, val_ids in kfold.split(dataset):
                kfold_split_ids.append((train_ids, val_ids))

            kfoldSplits.append(kfold_split_ids)

    (train_ids, val_ids) = kfoldSplits[phase - 1][fold]
    dataset = trainingImageDatasets[phase - 1]

    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)

    for i in range(phase - 1):
        (train_ids, val_ids) = kfoldSplits[i][fold]
        dataset = trainingImageDatasets[i]

        temp_subset = get_memory_subset(dataset, train_ids, MEMORY_SIZE - 1)
        train_subset = ConcatDataset([train_subset, temp_subset])

        temp_subset = get_memory_subset(dataset, val_ids, 1)
        val_subset = ConcatDataset([val_subset, temp_subset])

    return train_subset, val_subset
