#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Wrong format: use ./from_scratch.sh <#iterations>"
    exit -1
fi

cd From-Scratch/$1/

# Iterating over pruning rates
#for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do
for i in 45 50 55 60 65 70 75 80 85 90 95; do

    cd $i
    # Training from-scratch (random initial weights)
    ./darknet detector train dfire.data dfire.cfg -dont_show -map

    for j in $(seq 1000 1000 "$1"); do
       rm weights/dfire_$j.weights
    done

    # Returns to From-Scratch/$1
    cd ..

done
