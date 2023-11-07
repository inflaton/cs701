import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
input_size = 224

# Initialize transformations for data augmentation
# https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# preprocess_image = transforms.Compose(
#     [
#         transforms.RandomResizedCrop((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=45),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ]
# )

# copied from Jialong's code
preprocess_image = A.Compose(
    [
        A.SmallestMaxSize(max_size=input_size + 48),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=input_size, width=input_size),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
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
    2087,
]

NUM_PHASES = len(num_images_in_phase) - 2
MEMORY_SIZE = 5
NUM_CLASSES_IN_PHASE = 10


# Create Custom PyTorch Dataset
class CustomImageDataset(Dataset):
    def __init__(self, phase, transform=None, load_training_images_for_val=False):
        self.phase = phase
        self.transform = transform
        self.img_labels = []
        self.image_paths = []
        self.load_training_images_for_val = load_training_images_for_val

        if phase <= 0:
            self.load_val_test_images(phase < 0)
        else:
            self.load_training_images(phase)

    def load_training_images(self, phase):
        for i in range(phase):
            for j in range(NUM_CLASSES_IN_PHASE):
                label = i * NUM_CLASSES_IN_PHASE + j
                if i < phase - 1:
                    data_dir = f"data/Train/phase_{i + 1}/{label:03d}"
                    num_images = MEMORY_SIZE
                else:
                    data_dir = f"data/Train/phase_{i + 1}/{label:03d}"
                    num_images = num_images_in_phase[i - 1]

                for k in range(num_images):
                    image_filename = f"{k:03d}.jpg"
                    image_path = os.path.join(data_dir, image_filename)

                    # Check if the image file exists
                    if os.path.exists(image_path):
                        self.img_labels.append(label)
                        self.image_paths.append(image_path)

    def load_val_test_images(self, is_testing):
        data_dir = "data/Test" if is_testing else "data/Val"
        print("data_dir: ", data_dir)
        for k in range(num_images_in_phase[-1 if is_testing else -2]):
            image_filename = f"{k:04d}.jpg" if is_testing else f"{k:03d}.jpg"
            image_path = os.path.join(data_dir, image_filename)

            # Check if the image file exists
            if os.path.exists(image_path):
                self.img_labels.append(image_filename)
                self.image_paths.append(image_path)
            else:
                # print("not found: ", image_path)
                pass

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        image_path = self.image_paths[idx]
        # print("image_path: ", image_path)

        if (
            self.transform is None
            or self.transform == preprocess_val_image
            or self.phase <= 0
        ):
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            # use cv2 for albumentation
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=image)["image"]

        return img, label


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, n_classes, model_ver=1, used_for_icarl=False):
        super().__init__()
        self.model_ver = model_ver

        if model_ver == 1:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.resnet_fc_in_features = resnet.fc.in_features
        self.fc_for_classfication = self.new_fc_layer(n_classes)

        identity = Identity()
        resnet.fc = identity if used_for_icarl else self.fc_for_classfication

        self.n_classes = n_classes
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
        self.used_for_icarl = used_for_icarl

    def forward(self, x, features=False):
        output = self.base_model(x)

        if not features:
            if self.used_for_icarl:
                output = self.fc_for_classfication(output)
            else:
                output = self.sigm(output)

        return output

    def new_fc_layer(self, n_classes):
        if self.model_ver == 1:
            return nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(
                    in_features=self.resnet_fc_in_features, out_features=n_classes
                ),
            )
        return nn.Linear(in_features=self.resnet_fc_in_features, out_features=n_classes)

    def update_num_classes(self, n_classes):
        self.n_classes = n_classes

        fc = self.fc_for_classfication[-1]
        in_features = fc.in_features
        out_features = fc.out_features
        weight = fc.weight.data
        bias = fc.bias.data

        fc = torch.nn.Linear(in_features, n_classes)
        fc.weight.data[:out_features] = weight
        fc.bias.data[:out_features] = bias

        self.fc_for_classfication[-1] = fc


# save checkpoint function
def checkpoint_save(model, save_path, epoch, model_ver=1):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    if model_ver == 1:
        torch.save(model.state_dict(), f)
    else:
        torch.save(model.base_model.state_dict(), f)
    print("saved checkpoint:", f, flush=True)


def checkpoint_delete(save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    os.remove(f)


# save checkpoint function
def checkpoint_load(model, save_path, epoch, n_classes=0, model_ver=1):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)

    backup = None
    if n_classes > 0 and n_classes != model.n_classes:
        backup = model.n_classes
        model.update_num_classes(n_classes)

    if model_ver == 1:
        model.load_state_dict(torch.load(f))
    else:
        model.base_model.load_state_dict(torch.load(f))

    if backup is not None:
        model.update_num_classes(backup)

    print("loaded checkpoint:", f, flush=True)


# calculate metrics function
def calculate_metrics(pred, target):
    pred = [np.argmax(i) for i in pred]
    return {
        "weighted/f1": f1_score(y_true=target, y_pred=pred, average="weighted"),
    }


class TrainingImageDataset(Dataset):
    def __init__(self, phase, transform=None, load_training_images_for_val=False):
        self.phase = phase
        self.transform = transform
        self.img_labels = []
        self.image_paths = []
        self.load_training_images_for_val = load_training_images_for_val

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

        if (
            self.transform is None
            or self.load_training_images_for_val
            or self.transform == preprocess_val_image
        ):
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            # use cv2 for albumentation
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=image)["image"]

        return img, label


trainingImageDatasets = [
    TrainingImageDataset(phase + 1, transform=preprocess_image)
    for phase in range(NUM_PHASES)
]


def get_training_datasets(phase, prev_train_sets=[], prev_val_sets=[]):
    dataset = trainingImageDatasets[phase - 1]

    train_len = int(len(dataset) * 7 / 10)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_len, len(dataset) - train_len]
    )

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


def get_k_fold_training_datasets(phase, fold, use_memory=True):
    print(
        f"get_k_fold_training_datasets: phase={phase} fold={fold} use_memory={use_memory}"
    )
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

        if use_memory:
            temp_train_subset = get_memory_subset(dataset, train_ids, MEMORY_SIZE - 1)
            temp_val_subset = get_memory_subset(dataset, val_ids, 1)

        else:
            temp_train_subset = Subset(dataset, train_ids)
            temp_val_subset = Subset(dataset, val_ids)

        train_subset = ConcatDataset([train_subset, temp_train_subset])
        val_subset = ConcatDataset([val_subset, temp_val_subset])

    return train_subset, val_subset


def get_final_validation_dataset(phase, transform=preprocess_val_image):
    datasets = [
        TrainingImageDataset(
            i + 1, transform=transform, load_training_images_for_val=True
        )
        for i in range(phase)
    ]

    final_val_set = None
    for dataset in datasets:
        train_len = int(len(dataset) * 7 / 10)
        _, val_set = torch.utils.data.random_split(
            dataset, [train_len, len(dataset) - train_len]
        )
        if final_val_set is None:
            final_val_set = val_set
        else:
            final_val_set = ConcatDataset([final_val_set, val_set])

    return final_val_set


def get_icarl_batches(transform_train=None, transofrm_val=None):
    datasets = [
        TrainingImageDataset(i + 1, transform=transform_train)
        for i in range(NUM_PHASES)
    ]

    val_datasets = [
        TrainingImageDataset(i + 1, transform=transofrm_val) for i in range(NUM_PHASES)
    ]

    train_batches, test_batches = [], []

    for dataset, val_dataset in zip(datasets, val_datasets):
        train_ids, val_ids = torch.utils.data.random_split(
            range(len(dataset)), [0.8, 0.2]
        )

        train_set = Subset(dataset, train_ids)
        train_batches.append(train_set)

        val_set = Subset(val_dataset, val_ids)
        test_batches.append(val_set)

    return train_batches, test_batches
