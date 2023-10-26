# CS701 - Submission by Team 2 

This project contains all source files, as well as results/logs, for our submission. All sample commands below should work on Linux (including WSL2 on Windows) or Mac.

## How to run it

1. create a subfolder `data` and put train/val/test image folders under `data` folder.

```
ls data | grep "Train\|Val\|Test"
Test
Train
Val
```

2. install all depedencies:
```
pip install -r requirements.txt
```

3. to reproduce results sumbitted to the public leaderboard,
* train and validate model with the following command.
* model checkpoints for all phases are stored under subfolders `data/checkpoints_phase_{phase}`.
* all validation results are stored in `results` folder which is compressed into `validation.zip`.
```
./train.sh
```

4. to reproduce results sumbitted to the private leaderboard,
* run the following command which will load the checkpoints from step 3) and generate results
* all test results are stored in `results` folder which isompressed into `test.zip`. 
```
./test.sh
```