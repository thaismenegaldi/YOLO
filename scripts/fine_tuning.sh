#!/bin/bash

if [ "$#" -ne 3 ]
then
    echo "Wrong format: use ./fine_tuning.sh <#iterations> <pruning_technique> <cutting_layer>"
    exit -1
fi

cd Fine-Tuning/$1/$2/

# Iterating over pruning rates
for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do

    cd $i/
    # Generating pre-weights
    ./darknet partial dfire.cfg dfire.weights dfire$i.conv.$3 $3
    # Fine tuning
    ./darknet detector train dfire.data dfire.cfg dfire$i.conv.$3 -dont_show -map

    for j in $(seq 1000 1000 "$1"); do
        rm weights/dfire_$j.weights
    done

    # Returns to Fine-Tuning/<pruning_technique>/
    cd ..

done
