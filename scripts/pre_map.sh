#!/bin/bash

if [ "$#" -ne 2 ]
then
    echo "Wrong format: use ./pre_map.sh <#iterations> <pruning_technique>"
    exit -1
fi

cd Fine-Tuning/$1/$2/

# Iterating over pruning rates
for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do

    cd $i

    # Evaluating
    ./darknet detector map dfire.data dfire.cfg dfire.weights 

    # Returns to Fine-Tuning/<pruning_technique>/
    cd ..

done
