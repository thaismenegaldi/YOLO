#!/bin/bash

# Remove temporary files
rm -r ../__pycache__
rm ../train_batch0.jpg
rm ../test_batch0_gt.jpg
rm ../test_batch0_pred.jpg
rm ../results.png
rm ../results.txt
rm ../nohup.out
cd ../utils/
rm -r __pycache__
