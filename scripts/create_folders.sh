#!/bin/bash

if [ "$#" -ne 2 ]
then
    echo "Wrong format: use ./preparing_directories.sh <#iterations> <pruning_technique>"
    exit -1
fi

cd Models/$1
mkdir -p $2/
cd $2/

# Iterating over pruning rates
for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do

    mkdir -p $i/
    chmod 777 $i/
done
