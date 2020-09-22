#!/usr/bin/env bash

pip3 install -r requirements.txt
mkdir data
unzip -u preprocess/training_set_raw.zip -d preprocess
mv -f preprocess/training.1600000.processed.noemoticon.csv preprocess/training_set_raw.csv

echo 'running preprocess on haggle dataset and fetching validation set'
echo 'this may take a while...'
python3 ./preprocess/preprocess.py & python3 ./data_fetch/fetch.py
wait
echo 'setup finished, press any key to exit'
read
