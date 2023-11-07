import argparse
import cv2

import yaml
from pathlib import Path
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import zipfile
import random
import numpy as np
import timm

from utils import (
    checkpoint_load,
    checkpoint_save,
    preprocess_image,
)


class CustomDataset(Dataset):
    memory_buffer = {}  # set old memeory

    def __init__(self, root, phase, transform=None, is_val=False):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.img_names = []
        self.is_val = is_val

        if is_val:
            self.img_names = sorted(
                [name for name in os.listdir(root)], key=lambda x: int(x.split(".")[0])
            )
            self.imgs = [os.path.join(root, name) for name in self.img_names]
        else:
            for class_idx in range(phase * 10):
                class_folder = os.path.join(
                    root, f"phase_{(class_idx // 10) + 1}", f"{class_idx:03d}"
                )
                if os.path.exists(class_folder):
                    imgs_for_this_class = []  # path of certain class
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        imgs_for_this_class.append(img_path)
                        self.imgs.append(img_path)
                        self.labels.append(class_idx)

                    # update old memory
                    if class_idx not in self.memory_buffer:
                        self.memory_buffer[class_idx] = []
                    self.memory_buffer[class_idx].extend(imgs_for_this_class)
                    # no more than 5 in old memory
                    if len(self.memory_buffer[class_idx]) > 5:
                        self.memory_buffer[class_idx] = random.sample(
                            self.memory_buffer[class_idx], 5
                        )

            # combine memory with data
            for class_idx, img_paths in self.memory_buffer.items():
                self.imgs.extend(img_paths)
                self.labels.extend([class_idx] * len(img_paths))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        if self.is_val:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            # use cv2 for albumentation
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=image)["image"]
        if self.labels:
            label = self.labels[idx]
            return img, label
        else:
            return img, self.img_names[idx]


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.1, temperature=1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, teacher_outputs):
        if outputs.shape[1] != teacher_outputs.shape[1]:
            padding = torch.zeros(
                (teacher_outputs.shape[0], outputs.shape[1] - teacher_outputs.shape[1]),
                device=teacher_outputs.device,
            )
            teacher_outputs = torch.cat((teacher_outputs, padding), dim=1)

        loss = (1 - self.alpha) * self.criterion(outputs, labels)
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(outputs / self.temperature, dim=1),
            nn.functional.softmax(teacher_outputs / self.temperature, dim=1),
        )
        loss += self.alpha * distillation_loss
        return loss


def train_and_evaluate(
    num_epochs, train_dataset, val_dataset, model, teacher_model, phase, device
):
    SAVE_PATH = os.path.join(os.getcwd(), "data", f"checkpoints_phase_{phase}/")
    os.makedirs(SAVE_PATH, exist_ok=True)

    criterion = DistillationLoss() if teacher_model else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=44, gamma=0.1)

    train_set, val_set = random_split(train_dataset, [0.9, 0.1])
    print(f"training phase: {phase}")
    print(f"train_set len: {len(train_set)}")
    print(f"val_set len: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model.to(device)
    if teacher_model:
        teacher_model.to(device)

    highest_accuracy = 50.0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        print(
            f"Phase {phase}, Epoch {epoch}, Learning rate: {learning_rate:.10f}",
            flush=True,
        )

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss = criterion(outputs, labels, teacher_outputs)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # scheduler.step()
        epoch_loss = running_loss / len(train_set)
        print(
            f"\tLoss: {epoch_loss}",
            flush=True,
        )

        model.eval()
        with torch.no_grad():
            model_result = []
            all_labels = []

            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

            for my_images, my_labels in val_loader:
                all_labels.extend(my_labels.numpy())

                my_images = my_images.to(device, non_blocking=True)
                my_labels = my_labels.to(device, non_blocking=True, dtype=torch.int64)

                output = model(my_images)
                model_result.extend(output.cpu().numpy())

            acc1_cls = timm.utils.accuracy(
                torch.from_numpy(np.array(model_result)),
                torch.from_numpy(np.array(all_labels)),
                topk=(1,),
            )[0]

            print(
                f"\tAccuracy: {acc1_cls:.3f}%",
                flush=True,
            )
            if acc1_cls > highest_accuracy:
                highest_accuracy = acc1_cls
                best_epoch = epoch
                checkpoint_save(model, SAVE_PATH, epoch)

    checkpoint_load(model, SAVE_PATH, best_epoch)

    model.eval()
    predictions = []
    img_names = []
    with torch.no_grad():
        for inputs, names in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            img_names.extend(names)

    filename = f"results/result_{phase}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        for name, pred in zip(img_names, predictions):
            file.write(f"{name} {pred}\n")

    print(f"saved results to file: {filename}", flush=True)

    return model, highest_accuracy


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=100)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=128)

# Parse the arguments
args = parser.parse_args()

batch_size = args.batch
num_epochs = args.epochs

print(
    "epochs: ",
    num_epochs,
    "\nbatch: ",
    batch_size,
)

RANDOM_SEED = 193

# initialising seed for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
seeded_generator = torch.Generator().manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = None

config = None
filename = "configs-timm-x.yaml"
path = Path(filename)
if path.is_file():
    # Load the YAML file
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

phase_accuracy = []

for phase in range(1, 11):
    num_classes = phase * 10

    model = timm.create_model(
        "resnext50_32x4d.a2_in1k",
        pretrained=True,
        num_classes=num_classes,
    )
    data_config = timm.data.resolve_model_data_config(model)
    transform_for_training = preprocess_image
    # timm.data.create_transform(**data_config, is_training=True)

    train_dataset = CustomDataset(
        root="data/Train", phase=phase, transform=transform_for_training, is_val=False
    )

    transform_for_validation = timm.data.create_transform(
        **data_config, is_training=False
    )
    val_dataset = CustomDataset(
        root="data/Val", phase=phase, transform=transform_for_validation, is_val=True
    )

    if teacher_model:
        new_state_dict = model.state_dict()
        old_state_dict = {
            name: param
            for name, param in teacher_model.state_dict().items()
            if name in new_state_dict and new_state_dict[name].shape == param.shape
        }
        new_state_dict.update(old_state_dict)
        model.load_state_dict(new_state_dict, strict=False)

    model, highest_accuracy = train_and_evaluate(
        num_epochs if config is None else config["num_epoch"][f"phase_{phase}"],
        train_dataset,
        val_dataset,
        model,
        teacher_model,
        phase,
        device,
    )
    teacher_model = model
    phase_accuracy.append(highest_accuracy)

print("phase_accuracy: ", phase_accuracy)
print("final accuracy: ", phase_accuracy[-1])
print("avarage accuracy: ", np.mean(phase_accuracy))

# compress the results folder
filename = "validation.zip"
path = Path(filename)
if path.is_file():
    os.remove(filename)
with zipfile.ZipFile("validation.zip", "w") as zipf:
    for dirname, subdirs, files in os.walk("results"):
        zipf.write(dirname)
        for filename in files:
            zipf.write(os.path.join(dirname, filename))
