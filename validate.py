import gc
import os
from pathlib import Path
import re
import time
import zipfile
import numpy as np
import pandas as pd
import torch
import argparse

from utils import (
    NUM_CLASSES_IN_PHASE,
    NUM_PHASES,
    calculate_metrics,
    device,
    get_final_validation_dataset,
    preprocess_val_image,
    checkpoint_load,
    CustomImageDataset,
    NeuralNetwork,
)
import random
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=int, help="Model impl version", default=1)
parser.add_argument("-p", "--phase", type=int, help="Training phase", default=1)
parser.add_argument(
    "-c", "--checkpoint", type=int, help="Checkpoint to load", default=0
)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)
parser.add_argument(
    "-v", "--val_or_test", type=int, help="val_or_test: 0=validation 1=test", default=0
)

# Parse the arguments
args = parser.parse_args()

model_ver = args.model
batch_size = args.batch
checkpoint = args.checkpoint
phase = args.phase
val_or_test = args.val_or_test
num_classes = NUM_CLASSES_IN_PHASE * phase

print(
    "model: ",
    model_ver,
    "\nclasses: ",
    num_classes,
    "\ncheckpoint: ",
    checkpoint,
    "\nbatch: ",
    batch_size,
    "\nphase: ",
    phase,
)

RANDOM_SEED = 193

# initialising seed for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
seeded_generator = torch.Generator().manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

start_time = time.time()

RESULT_PATH = os.path.join(os.getcwd(), "results/")
SAVE_PATH = os.path.join(os.getcwd(), "data", f"checkpoints_phase_{phase}/")

# make checkpoints and results dir
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# initialise model instance
# Initialize the model for this run
model = NeuralNetwork(num_classes, model_ver=model_ver)

for dirname, subdirs, files in os.walk(SAVE_PATH):
    filename = files[0]
    checkpoint = int(re.split("[-.]", filename)[-2])
    checkpoint_load(model, SAVE_PATH, checkpoint, model_ver=model_ver)

# transfer over to gpu
model = model.to(device)

# Start validations
torch.cuda.empty_cache()
gc.collect()

model.eval()
with torch.no_grad():
    model_result = []
    image_filenames = []
    val_set = CustomImageDataset(val_or_test, transform=preprocess_val_image)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    for inputs, labels in testloader:
        inputs = inputs.to(device)
        model_batch_result = model(inputs)
        model_result.extend(model_batch_result.cpu().numpy())
        image_filenames.extend(labels)

    pred = [np.argmax(i) for i in model_result]

    result_filename = f"{RESULT_PATH}result_{phase}.txt"
    with open(result_filename, "w") as file:
        count = 0
        for f, p in zip(image_filenames, pred):
            file.write(f"{f} {p}\n")
            count += 1

    print(f"{count} results saved to: {result_filename}")

    model_result = []
    total_targets = []
    test_loader = torch.utils.data.DataLoader(
        get_final_validation_dataset(phase), batch_size=batch_size
    )
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model_batch_result = model(inputs)

        model_result.extend(model_batch_result.cpu().numpy())
        total_targets.extend(targets.cpu().numpy())

    result = calculate_metrics(np.array(model_result), np.array(total_targets))
    accuracy = result["weighted/f1"]
    print(
        "phase:{:2d} - final accuracy:{:.3f}".format(
            phase,
            accuracy,
        ),
        flush=True,
    )

    filename = f"logs/validation.csv"

    path = Path(filename)
    file_exists = path.is_file()

    file = open(filename, "a")
    if not file_exists:
        file.write(f"phase,accuracy\n")

    file.write(f"{phase},{accuracy:.3f}\n")

# compress the results folder
filename = "result.zip"
path = Path(filename)
if path.is_file():
    os.remove(filename)
with zipfile.ZipFile(filename, "w") as zipf:
    for dirname, subdirs, files in os.walk("results"):
        zipf.write(dirname)
        for filename in files:
            zipf.write(os.path.join(dirname, filename))


# Calculate time elapsed
end_time = time.time()
time_difference = end_time - start_time
hours, rest = divmod(time_difference, 3600)
minutes, seconds = divmod(rest, 60)
print(
    "Validation is completed in {} hours, {} minutes, {:.3f} seconds".format(
        hours, minutes, seconds
    )
)
