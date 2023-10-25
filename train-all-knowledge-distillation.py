import gc
import os
import re
import time

import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
import argparse

from utils import (
    NUM_CLASSES_IN_PHASE,
    MEMORY_SIZE,
    NUM_PHASES,
    device,
    checkpoint_save,
    checkpoint_load,
    checkpoint_delete,
    calculate_metrics,
    get_k_fold_training_datasets,
    NeuralNetwork,
)
import random
import warnings

warnings.filterwarnings("ignore")

# Load the YAML file
with open("configs.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, help="Model impl version", default=1)
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=20)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)
parser.add_argument("-s", "--baseline", type=int, help="Training iteration", default=0)

# Parse the arguments
args = parser.parse_args()

model_ver = args.model
batch_size = args.batch
num_epochs = args.epochs
baseline = args.baseline == 1

print(
    "model: ",
    model_ver,
    "\nepochs: ",
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

# SETTING CONSTANTS HERE
LR = 1e-4

start_time = time.time()


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.1, temperature=1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, teacher_outputs=None):
        if teacher_outputs is None:
            return self.criterion(outputs, labels)

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


def train_and_evaluate(model, teacher_model, phase, device, all_phases):
    num_classes = NUM_CLASSES_IN_PHASE * phase

    # transfer over to gpu
    model = model.to(device)
    if teacher_model:
        teacher_model.to(device)

    num_epochs = config["num_epoch"][f"phase_{phase}"]

    for fold in range(MEMORY_SIZE):
        print(
            "phase: ",
            phase,
            "\nclasses: ",
            num_classes,
            "\nfold: ",
            fold + 1,
            "\nnum_epochs: ",
            num_epochs,
        )

        # set data path
        SAVE_PATH = os.path.join(os.getcwd(), "data", f"checkpoints_phase_{phase}/")
        os.makedirs(SAVE_PATH, exist_ok=True)

        model.train()
        running_loss = 0.0

        criterion = DistillationLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Start training
        torch.cuda.empty_cache()
        gc.collect()

        train_set, val_set = get_k_fold_training_datasets(phase, fold, not baseline)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

        phases = []
        folds = []
        epochs = []
        train_losses = []
        val_losses = []
        accuracies = []
        learning_rates = []

        for epoch in range(0, num_epochs):
            print(f"Starting epoch {epoch+1}")

            phases.append(phase)
            folds.append(fold + 1)
            epochs.append(epoch + 1)

            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if teacher_model:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    loss = criterion(outputs, targets, teacher_outputs)
                else:
                    loss = criterion(outputs, targets)

                # backward + optimize
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
            learning_rates.append(learning_rate)
            print(
                f"phase:{phase:2d} fold:{fold + 1:2d} epoch:{epoch + 1:2d} - learning rate:{learning_rate:.7f}"
            )
            scheduler.step()

            epoch_loss = running_loss / len(train_loader.dataset)
            print(
                f"phase:{phase:2d} fold:{fold + 1:2d} epoch:{epoch + 1:2d} - train loss:{epoch_loss:.3f}"
            )
            train_losses.append(epoch_loss)

            # Validation for this fold
            val_loss = []
            model.eval()
            with torch.no_grad():
                model_result = []
                total_targets = []
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    model_batch_result = model(inputs)

                    model_result.extend(model_batch_result.cpu().numpy())
                    total_targets.extend(targets.cpu().numpy())

                    loss = criterion(model_batch_result, targets)
                    batch_loss_value = loss.item()
                    val_loss.append(batch_loss_value)

                result = calculate_metrics(
                    np.array(model_result), np.array(total_targets)
                )
                accuracy = result["weighted/f1"]

                # Print statistics
                loss_value = np.mean(val_loss)
                print(
                    "phase:{:2d} fold:{:2d} epoch:{:2d} - val loss:{:.3f}".format(
                        phase, fold + 1, epoch + 1, loss_value
                    )
                )
                print(
                    "phase:{:2d} fold:{:2d} epoch:{:2d} - accuracy:{:.3f}".format(
                        phase,
                        fold + 1,
                        epoch + 1,
                        accuracy,
                    ),
                    flush=True,
                )
                val_losses.append(loss_value)
                accuracies.append(accuracy)

            checkpoint_save(model, SAVE_PATH, epoch)

            if 1 - accuracy < 1e-8:
                break

        df = pd.DataFrame(
            {
                "phase": phases,
                "fold": folds,
                "epoch": epochs,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "accuracy": accuracies,
                "learning_rate": learning_rates,
            }
        )
        df.to_csv(f"logs/phase_{phase}.csv", index=False)

        all_phases = pd.concat([all_phases, df])

        top_checkpoint = df["accuracy"].idxmax()
        print(
            "*** phase:{:2d} fold:{:2d} epoch:{:2d} - top accuracy:{:.3f}".format(
                phase,
                fold + 1,
                top_checkpoint + 1,
                df["accuracy"][top_checkpoint],
            ),
            flush=True,
        )

        checkpoint_load(model, SAVE_PATH, top_checkpoint)

        for _, _, files in os.walk(SAVE_PATH):
            for filename in files:
                checkpoint = int(re.split("[-.]", filename)[-2])
                if top_checkpoint != checkpoint:
                    checkpoint_delete(SAVE_PATH, checkpoint)

    return model, all_phases


teacher_model = None
all_phases = pd.DataFrame(
    {
        "phase": [],
        "fold": [],
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "learning_rate": [],
    }
)

for i in range(NUM_PHASES):
    phase = i + 1
    num_classes = NUM_CLASSES_IN_PHASE * phase
    model = NeuralNetwork(num_classes, model_ver=model_ver)

    if teacher_model:
        new_state_dict = model.state_dict()
        old_state_dict = {
            name: param
            for name, param in teacher_model.state_dict().items()
            if name in new_state_dict and new_state_dict[name].shape == param.shape
        }
        new_state_dict.update(old_state_dict)
        model.load_state_dict(new_state_dict, strict=False)

    teacher_model, all_phases = train_and_evaluate(
        model, teacher_model, phase, device, all_phases
    )

all_phases.to_csv("logs/all_phases.csv", index=False)

# Calculate time elapsed
end_time = time.time()
time_difference = end_time - start_time
hours, rest = divmod(time_difference, 3600)
minutes, seconds = divmod(rest, 60)
print(
    "Training is completed in {} hours, {} minutes, {:.3f} seconds".format(
        hours, minutes, seconds
    )
)
