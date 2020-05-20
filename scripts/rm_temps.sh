#!/bin/bash

# Remove temporary files
rm -r ../__pycache__
rm ../train_batch0.png
rm ../test_batch0.png
rm ../results.png
rm ../results.txt
rm ../nohup.out
cd ../utils/
rm -r __pycache__
