#! /bin/bash

echo "Generating 4 datasets with 1000 signals each and \
sensitivity 5.0, 10.0, 15.0 and 20.0 respectively"

python generate_data.py --sensitivity 5.0 --num_signals 1000
python generate_data.py --sensitivity 10.0 --num_signals 1000
python generate_data.py --sensitivity 15.0 --num_signals 1000
python generate_data.py --sensitivity 20.0 --num_signals 1000