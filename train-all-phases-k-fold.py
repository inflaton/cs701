import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
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

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=20)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)

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


# SETTING CONSTANTS HERE
LR = 1e-4

# https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
LR = 1e-3

start_time = time.time()

model = None

all_phases = pd.DataFrame(
    {"phase": [], "fold": [], "epoch": [], "loss": [], "accuracy": []}
)

for i in range(NUM_PHASES):
    phase = i + 1

    num_classes = NUM_CLASSES_IN_PHASE * phase

    for fold in range(MEMORY_SIZE):
        print(
            "phase: ",
            phase,
            "\nclasses: ",
            num_classes,
            "\nfold: ",
            fold + 1,
        )

        # initialise model instance
        # Initialize the model for this run
        if model is None:
            model = NeuralNetwork(num_classes)
        else:
            checkpoint_load(model, SAVE_PATH, top_checkpoint + 1)
            if fold == 0:
                model.update_num_classes(num_classes)

        # set data path
        SAVE_PATH = os.path.join(os.getcwd(), "data", f"checkpoints_phase_{phase}/")
        os.makedirs(SAVE_PATH, exist_ok=True)

        model.train()

        # transfer over to gpu
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Start training
        torch.cuda.empty_cache()
        gc.collect()

        train_set, val_set = get_k_fold_training_datasets(phase, fold)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

        phases = []
        folds = []
        epochs = []
        losses = []
        accuracies = []

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
                loss = criterion(outputs, targets)

                # backward + optimize
                loss.backward()
                optimizer.step()

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
                    "phase:{:2d} fold:{:2d} epoch:{:2d} - loss:{:.3f}".format(
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
                losses.append(loss_value)
                accuracies.append(accuracy)

            checkpoint_save(model, SAVE_PATH, epoch + 1)

            if 1 - accuracy < 1e-8:
                break

        df = pd.DataFrame(
            {
                "phase": phases,
                "fold": folds,
                "epoch": epochs,
                "loss": losses,
                "accuracy": accuracies,
            }
        )
        df.to_csv(f"logs/phase_{phase}.csv", index=False)

        all_phases = pd.concat([all_phases, df])

        top_checkpoint = df["accuracy"].idxmax()
        for epoch in range(len(epochs)):
            if epoch != top_checkpoint:
                checkpoint_delete(SAVE_PATH, epoch + 1)

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
