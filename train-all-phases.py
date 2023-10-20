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
    device,
    checkpoint_save,
    checkpoint_load,
    checkpoint_delete,
    calculate_metrics,
    get_training_datasets,
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
SAVE_FREQ = 1  # save checkpoint frequency (epochs)

start_time = time.time()

model = None

train_sets = []
val_sets = []

for i in range(10):
    phase = i + 1

    num_classes = NUM_CLASSES_IN_PHASE * phase

    print(
        "phase: ",
        phase,
        "\nclasses: ",
        num_classes,
    )

    # initialise model instance
    # Initialize the model for this run
    if model is None:
        model = NeuralNetwork(num_classes)
    else:
        checkpoint_load(model, SAVE_PATH, top_checkpoint + 1)
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

    epoch = 0
    iteration = 0

    # initialise dataset and dataloader instance
    train_set, val_set = get_training_datasets(phase, train_sets, val_sets)

    train_sets.append(train_set)
    val_sets.append(val_set)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    epochs = []
    losses = []
    accuracies = []

    for epoch in range(0, num_epochs):
        print(f"Starting epoch {epoch+1}")

        running_loss = []

        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_loss_value = loss.item()
            running_loss.append(batch_loss_value)

            # backward + optimize
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            model_result = []
            total_targets = []
            for inputs, targets in testloader:
                inputs = inputs.to(device)
                model_batch_result = model(inputs)
                model_result.extend(model_batch_result.cpu().numpy())
                total_targets.extend(targets.cpu().numpy())

            result = calculate_metrics(np.array(model_result), np.array(total_targets))

            # Print statistics
            loss_value = np.mean(running_loss)
            print("epoch:{:2d} train: loss:{:.3f}".format(epoch + 1, loss_value))
            print(
                "epoch:{:2d} test: "
                "weighted f1 {:.3f}".format(
                    epoch + 1,
                    result["weighted/f1"],
                ),
                flush=True,
            )
            epochs.append(epoch + 1)
            losses.append(loss_value)
            accuracies.append(result["weighted/f1"])

        if epoch % SAVE_FREQ == 0:
            checkpoint_save(model, SAVE_PATH, epoch + 1)

    df = pd.DataFrame({"epoch": epochs, "loss": losses, "accuracy": accuracies})
    df.to_csv(f"logs/phase_{phase}.csv", index=False)

    top_checkpoint = df["accuracy"].idxmax()
    for epoch in range(0, num_epochs):
        if epoch != top_checkpoint:
            checkpoint_delete(SAVE_PATH, epoch + 1)


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
