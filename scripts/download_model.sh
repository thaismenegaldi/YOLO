#!/bin/bash

if [ "$#" -ne 2 ]
then
    echo "Wrong format: use ./download_models.sh <pruning_technique> <pruning_rate>"
    exit -1
fi

cd Models/
mkdir -p $1
cd $1/
mkdir -p $2
cd ..
cd ..

path=Models/$1/$2

# Download .weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=WEIGHTS_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=WEIGHTS_ID" -O $path/dfire.weights && rm -rf /tmp/cookies.txt

# Download .cfg
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=CFG_ID' -O $path/dfire.cfg
