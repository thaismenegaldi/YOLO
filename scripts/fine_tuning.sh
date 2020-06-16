#!/bin/bash

if [ "$#" -ne 3 ]
then
    echo "Wrong format: use ./fine_tuning.sh <pruning_technique> <pruning_rate> <cutting_layer> "
    exit -1
fi

cd Fine-Tuning/$1/$2/

# Generating pre-weights
./darknet partial dfire.cfg dfire.weights dfire.conv.$3 $3

# Fine tuning
./darknet detector train dfire.data dfire.cfg dfire.conv.$3 -dont_show -map
