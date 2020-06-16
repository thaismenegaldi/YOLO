#!/bin/bash

if [ "$#" -ne 2 ]
then
    echo "Wrong format: use ./create_folder.sh <pruning_technique> <pruning_rate>"
    exit -1
fi

path=$1/$2

if [ ! -e Standard/data/dfire_train.txt ]; then
    echo "Error: file train.txt does not exist"
    exit -1
elif [ ! -e Standard/data/dfire_valid.txt ]; then
    echo "Error: file valid.txt does not exist"
    exit -1
elif [ ! -e Standard/data/dfire.names ]; then
    echo "Error: file .names does not exist"
    exit -1
elif [ ! -e Standard/dfire.data ]; then
    echo "Error: file .data does not exist"
    exit -1
elif [ ! -e Models/$path/dfire.weights ]; then
    echo "Error: file .weights does not exist"
    exit -1
elif [ ! -e Models/$path/dfire.cfg ]; then
    echo "Error: file .cfg does not exist"
    exit -1
fi

cd Fine-Tuning/
mkdir -p $1/
cd $1/
mkdir $2/
cd ..
cd ..

cp -r Standard/data Fine-Tuning/$path/
cp Standard/dfire.data Fine-Tuning/$path/
cp Standard/darknet Fine-Tuning/$path/
cp Models/$path/dfire.weights Fine-Tuning/$path/
cp Models/$path/dfire.cfg Fine-Tuning/$path/

cd Fine-Tuning/$path/
mkdir weights

chmod +x darknet


