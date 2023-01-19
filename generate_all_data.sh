#! /bin/bash

echo "Generating 5 datasets with 2000 signals each and \
sensitivity 5.0, 10.0, 15.0, 20.0 and 25.0 respectively"

python generate_data.py --sensitivity 5.0 --num_signals 2000
python generate_data.py --sensitivity 10.0 --num_signals 2000
python generate_data.py --sensitivity 15.0 --num_signals 2000
python generate_data.py --sensitivity 20.0 --num_signals 2000
python generate_data.py --sensitivity 25.0 --num_signals 2000
