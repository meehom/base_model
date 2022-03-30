#!/bin/bash

raw_data_dir="data_batching/raw_data/"
target_dir="data_batching/clean_data/"
data_set_dir = "data_batching/clean_data/trainingSet.npy"

# log parameters
logs_dir="./log"
logs_name="./log/run_info.log"
# log_dir check
if [ ! -d $logs_dir ];then
   mkdir -p $logs_dir
fi

python data_process.py raw_data_dir target_dir
python train.py data_set_dir
python inference.py target_dir > $logs_name
