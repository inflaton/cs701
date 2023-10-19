import gc
import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import f1_score
import argparse

from utils import (
    NUM_CLASSES_IN_PHASE,
    device,
    preprocess_image,
    checkpoint_save,
    checkpoint_load,
    calculate_metrics,
    CustomImageDataset,
    NeuralNetwork,
)
import random
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--phase", type=int, help="Training phase", default=1)
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=20)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)
parser.add_argument(
    "-c", "--checkpoint", type=int, help="Checkpoint to load", default=0
)

# Parse the arguments
args = parser.parse_args()

batch_size = args.batch
num_epochs = args.epochs
phase = args.phase
checkpoint = args.checkpoint
num_classes = NUM_CLASSES_IN_PHASE * phase

print(
    "classes: ",
    num_classes,
    "\nepochs: ",
    num_epochs,
    "\nbatch: ",
    batch_size,
    "\nphase: ",
    phase,
    "\ncheckpoint: ",
    checkpoint,
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

# set data path
SAVE_PATH = os.path.join(os.getcwd(), "data", f"checkpoints_phase_{phase}/")

# make checkpoint and log dir
os.makedirs(SAVE_PATH, exist_ok=True)

# initialise model instance
# Initialize the model for this run
model = NeuralNetwork(num_classes)

if checkpoint > 0:
    path = os.path.join(os.getcwd(), "data", f"checkpoints_phase_{phase-1}/")
    checkpoint_load(model, path, checkpoint, num_classes - NUM_CLASSES_IN_PHASE)


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
dataset = CustomImageDataset(phase, transform=preprocess_image)

train_len = int(len(dataset) * 7 / 10)
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_len, len(dataset) - train_len]
)

# Define data loaders for training and testing data in this fold
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
running_loss = []

for epoch in range(0, num_epochs):
    print(f"Starting epoch {epoch+1}")

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
    if epoch % SAVE_FREQ == 0:
        checkpoint_save(model, SAVE_PATH, epoch + 1)

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
