#!/bin/bash

# Darknet53 weights (first 75 layers only)
wget -c https://pjreddie.com/media/files/darknet53.conv.74
mv darknet53.conv.74 ../weights/darknet53.conv.74

# DFire config file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qO2h0rh_N5A7xTsO8ZwW0jZe1FniRaQQ' -O dfire.cfg
mv dfire.cfg ../cfg/dfire.cfg

# DFire data file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rDjIHumG0e_4mA_cqMbwaSXR0I8VZkga' -O dfire.data
mv dfire.data ../data/dfire.data

# DFire names file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KWeG7RqqLEAtZhfFPCElkzmEeZNfyz0I' -O dfire.names
mv dfire.names ../data/dfire.names

# DFire train file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Dz9gNPYhuM48PQi_ItvZBy8e1PK5M_zm' -O dfire_train.txt
mv dfire_train.txt ../data/dfire_train.txt

# DFire valid file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wu6jMW4qFwUf_WPhR4zgIwEGSNvtOr76' -O dfire_valid.txt
mv dfire_valid.txt ../data/dfire_valid.txt